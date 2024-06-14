import os
import cv2
import torch
import torch.utils.data
import torch.utils.checkpoint
from tqdm import tqdm
from typing import Optional
from dataset import SimpleDataset
from omegaconf import OmegaConf
from typing import Optional, Dict
from torch.cuda.amp import autocast
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from utils.util import get_time_string, get_function_args
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from model.transformer_sd3 import SD3Transformer2DModel
from model.pipeline import StableDiffusion3Pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)

class SampleLogger:
    def __init__(
        self,
        logdir: str,
        subdir: str = "sample",
        num_samples_per_prompt: int = 1,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
    ) -> None:
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.num_sample_per_prompt = num_samples_per_prompt
        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir)
        
    def log_sample_images(
        self, batch, pipeline: StableDiffusion3Pipeline, device: torch.device, step: int
    ):
        sample_seeds = torch.randint(0, 100000, (self.num_sample_per_prompt,))
        sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds
        self.prompts = batch["prompt"]
        for idx, prompt in enumerate(tqdm(self.prompts, desc="Generating sample images")):
            image = batch["image"][idx, :, :, :].unsqueeze(0)
            image = image.to(device=device)
            generator = []
            for seed in self.sample_seeds:
                generator_temp = torch.Generator(device=device)
                generator_temp.manual_seed(seed)
                generator.append(generator_temp)    
            sequence = pipeline(
                prompt,
                height=image.shape[2],
                width=image.shape[3],
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_images_per_prompt=self.num_sample_per_prompt,
            ).images

            image = (image + 1.) / 2. # for visualization
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite(os.path.join(self.logdir, f"{step}_{idx}_{seed}.png"), image[:, :, ::-1] * 255)
            with open(os.path.join(self.logdir, f"{step}_{idx}_{seed}" + '.txt'), 'a') as f:
                f.write(batch['prompt'][idx])
            for i, img in enumerate(sequence):
                img.save(os.path.join(self.logdir, f"{step}_{idx}_{sample_seeds[i]}_output.png"))
            
            
def train(
    pretrained_model_path: str,
    logdir: str,
    train_steps: int = 5000,
    validation_steps: int = 1000,
    validation_sample_logger: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1, # important hyper-parameter
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    train_batch_size: int = 1,
    val_batch_size: int = 1,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_warmup_steps: int = 0,
    use_8bit_adam: bool = True,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    checkpointing_steps: int = 10000,
):
    
    args = get_function_args()
    time_string = get_time_string()
    logdir += f"_{time_string}"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))

    if seed is not None:
        set_seed(seed)

    accelerator = Accelerator(mixed_precision=mixed_precision)

    transformer =  SD3Transformer2DModel.from_pretrained(pretrained_model_path, subfolder="transformer")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2")
    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2", use_fast=False)
    text_encoder_3 = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder_3")
    tokenizer_3 = T5TokenizerFast.from_pretrained(pretrained_model_path, subfolder="tokenizer_3")
    
    pipeline = StableDiffusion3Pipeline(
        transformer = transformer,
        scheduler = scheduler,
        vae = vae,
        text_encoder = text_encoder, 
        tokenizer = tokenizer,
        text_encoder_2 = text_encoder_2,
        tokenizer_2 = tokenizer_2,
        text_encoder_3 = text_encoder_3,
        tokenizer_3 = tokenizer_3,
    )
    
    pipeline.set_progress_bar_config(disable=True)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder_3.requires_grad_(False)
    transformer.requires_grad_(False)
    
    trainable_modules = ("add_q_proj", "add_k_proj", "add_v_proj")
    # trainable_modules = ("proj_out")
    for name, module in transformer.named_modules():
        if name.endswith(trainable_modules):
            for params in module.parameters():
                params.requires_grad = True
        # for params in module.parameters():
        #     params.requires_grad = True
    
    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
    
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = transformer.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_dataset = SimpleDataset(root="./", mode='train')
    val_dataset = SimpleDataset(root="./", mode='test')
    
    print(train_dataset.__len__())
    print(val_dataset.__len__())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=train_steps * gradient_accumulation_steps,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SimpleSDM-3")
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    text_encoder_3.to(accelerator.device, dtype=weight_dtype)
    
    step = 0

    if validation_sample_logger is not None and accelerator.is_main_process:
        validation_sample_logger = SampleLogger(**validation_sample_logger, logdir=logdir)

    progress_bar = tqdm(range(step, train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)
    val_data_yielder = make_data_yielder(val_dataloader)

    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    while step < train_steps:
        batch = next(train_data_yielder)
        
        vae.eval()
        text_encoder.eval()    
        text_encoder_2.eval()
        text_encoder_3.eval()
        transformer.train()
        
        image = batch["image"].to(dtype=weight_dtype)
        prompt = batch["prompt"]
        
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(prompt=prompt, prompt_2=prompt, prompt_3=prompt)
        
        b, c, h, w = image.shape

        latents = vae.encode(image).latent_dist.sample()
        latents = latents * vae.scaling_factor
        # Sample noise that we'll add
        noise = torch.randn_like(latents)
        # Sample a random timestep for each image
        indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,))
        timestep = noise_scheduler.timesteps[indices].to(device=latents.device)
        # Add noise according to flow matching.
        sigmas = get_sigmas(timestep, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_latent = sigmas * noise + (1.0 - sigmas) * latents
        
        # Predict the noise residual
        model_pred = transformer(hidden_states=noisy_latent, timestep=timestep, encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds).sample
        
        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_latent
        
        weighting = (sigmas**-2.0).float()
        # if args.weighting_scheme == "sigma_sqrt":
        #     weighting = (sigmas**-2.0).float()
        # elif args.weighting_scheme == "logit_normal":
        #     # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        #     u = torch.normal(mean=args.logit_mean, std=args.logit_std, size=(bsz,), device=accelerator.device)
        #     weighting = torch.nn.functional.sigmoid(u)
        # elif args.weighting_scheme == "mode":
        #     # See sec 3.1 in the SD3 paper (20).
        #     u = torch.rand(size=(bsz,), device=accelerator.device)
        #     weighting = 1 - u - args.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)

        target = latents
        
        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        
        # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(transformer.parameters(), max_grad_norm)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            step += 1
            if accelerator.is_main_process:
                if validation_sample_logger is not None and step % validation_steps == 0:
                    transformer.eval()
                    val_batch = next(val_data_yielder)
                    with autocast():
                        validation_sample_logger.log_sample_images(
                            batch = val_batch,
                            pipeline=pipeline,
                            device=accelerator.device,
                            step=step,
                        )
                if step % checkpointing_steps == 0:
                    pipeline_save = StableDiffusion3Pipeline(
                        transformer=accelerator.unwrap_model(transformer),
                        vae=vae,
                        scheduler=scheduler,
                        text_encoder=text_encoder,
                        text_encoder_2=text_encoder_2,
                        text_encoder_3=text_encoder_3,
                        tokenizer=tokenizer,
                        tokenizer_2=tokenizer_2,
                        tokenizer_3=tokenizer_3
                    )
                    checkpoint_save_path = os.path.join(logdir, f"checkpoint_{step}")
                    pipeline_save.save_pretrained(checkpoint_save_path)

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)
    accelerator.end_training()


if __name__ == "__main__":
    config = "./config.yml"
    train(**OmegaConf.load(config))

# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --multi_gpu train.py