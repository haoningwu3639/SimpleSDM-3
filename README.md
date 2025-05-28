# SimpleSDM-3
This repository contains a simple and flexible PyTorch implementation of StableDiffusion-3 based on diffusers.
The main purpose is to make it easier for generative model researchers to do DIY design and fine-tuning based on the powerful SDM-3 model.

<div align="center">
   <img src="example/example.png">
</div>

## Limitations
- Please note that at fp16 precision, the inference of this model requires approximately 20G of GPU memory. Training only all add_q, add_k, and add_v layers requires about 29G of GPU memory, and training all parameters requires about 37G of GPU memory.

## Prepartion
- You should download the diffusers version checkpoints of SDM-3, from [SDM-3-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers), including scheduler, text_encoder(s), tokenizer(s), transformer, and vae. Then put it in the ckpt folder.
- You can also download the model in Python script (Note: You should login with your HuggingFace token first.):

```
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="stabilityai/stable-diffusion-3-medium-diffusers", local_dir="./ckpt")
```

- If you cannot access Huggingface, you can use [hf-mirror](https://hf-mirror.com/) to download models.

```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download stabilityai/stable-diffusion-3-medium-diffusers --local-dir ckpt --local-dir-use-symlinks False
```

## Requirements
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.3](https://pytorch.org/)
- diffusers == 0.29.0
- accelerate == 0.31.0
- transformers == 4.41.2
- bitsandbytes == 0.43.1

A suitable [conda](https://conda.io/) environment named `sdm3` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate sdm3
```

## Dataset Preparation
- You need write a DataLoader suitable for your own Dataset, because we just provide a simple example to test the code.

## Training
```
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --multi_gpu train.py
```

## Inference
```
CUDA_VISIBLE_DEVICES=0 python inference.py --prompt "A cat is running in the rain."
```

## Acknowledgements
Many thanks to the checkpoint from [SDM-3](https://huggingface.co/stabilityai/stable-diffusion-3-medium/) and code bases from [diffusers](https://github.com/huggingface/diffusers/) and [SimpleSDM](https://github.com/haoningwu3639/SimpleSDM/).
