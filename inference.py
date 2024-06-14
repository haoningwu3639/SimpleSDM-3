import os
import random
import argparse
import torch
import torch.utils.data
import torch.utils.checkpoint
from typing import Optional
from accelerate import Accelerator
from accelerate.logging import get_logger

from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders import AutoencoderKL

from model.transformer_sd3 import SD3Transformer2DModel
from model.pipeline import StableDiffusion3Pipeline

logger = get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--logdir', default="./inference/", type=str)
    parser.add_argument('--ckpt', default='./ckpt/', type=str)
    parser.add_argument('--prompt', default="A black cat is running in the rain.", type=str)    
    parser.add_argument('--num_inference_steps', default=28, type=int)
    parser.add_argument('--guidance_scale', default=7.0, type=float)
    return parser

def test(
    pretrained_model_path: str,
    logdir: str,
    prompt: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    mixed_precision: Optional[str] = "fp16"   # "fp16"
):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        
    accelerator = Accelerator(mixed_precision=mixed_precision)

    transformer =  SD3Transformer2DModel.from_pretrained(pretrained_model_path, subfolder="transformer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2")
    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2")
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
    
    # Removing the memory-intensive 4.7B parameter T5-XXL text encoder during inference can significantly decrease the memory requirements for SD3 with only a slight loss in performance.
    # pipeline = StableDiffusion3Pipeline(
    #     transformer = transformer,
    #     scheduler = scheduler,
    #     vae = vae,
    #     text_encoder = text_encoder, 
    #     tokenizer = tokenizer,
    #     text_encoder_2 = text_encoder_2,
    #     tokenizer_2 = tokenizer_2,
    #     text_encoder_3 = None,
    #     tokenizer_3 = None,
    # )
    
    transformer, pipeline = accelerator.prepare(transformer, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    text_encoder_3.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("SimpleSDM-3")

    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    text_encoder_3.eval()
    transformer.eval()
    
    sample_seed = random.randint(0, 100000)
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(sample_seed)
    
    pipeline.enable_model_cpu_offload()
    
    output = pipeline(
        prompt = prompt,
        height = 1024,
        width = 1024,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
    )

    output_image = output.images[0] # PIL Image here
    output_image.save(os.path.join(logdir, f"{prompt}.png"))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pretrained_model_path = args.ckpt
    logdir = args.logdir
    prompt = args.prompt
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    mixed_precision = "no" # "fp16",
    test(pretrained_model_path, logdir, prompt, num_inference_steps, guidance_scale, mixed_precision)

# CUDA_VISIBLE_DEVICES=0 python inference.py --prompt "A cat is running in the rain."