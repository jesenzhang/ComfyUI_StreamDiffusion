# ComfyUI_StreamDiffusion

# This is a simple implementation StreamDiffusion<a href="https://github.com/cumulo-autumn/StreamDiffusion.git" > for ComfyUI


# StreamDiffusion: A Pipeline-Level Solution for Real-Time Interactive Generation

**Authors:** [Akio Kodaira](https://www.linkedin.com/in/akio-kodaira-1a7b98252/), [Chenfeng Xu](https://www.chenfengx.com/), Toshiki Hazama, [Takanori Yoshimoto](https://twitter.com/__ramu0e__), [Kohei Ohno](https://www.linkedin.com/in/kohei--ohno/), [Shogo Mitsuhori](https://me.ddpn.world/), [Soichi Sugano](https://twitter.com/toni_nimono), [Hanying Cho](https://twitter.com/hanyingcl), [Zhijian Liu](https://zhijianliu.com/), [Kurt Keutzer](https://scholar.google.com/citations?hl=en&user=ID9QePIAAAAJ)

StreamDiffusion is an innovative diffusion pipeline designed for real-time interactive generation. It introduces significant performance enhancements to current diffusion-based image generation techniques.

[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2312.12491)
[![Hugging Face Papers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-papers-yellow)](https://huggingface.co/papers/2312.12491)

# Simple Use
you can download the workflow image below , and import into ComfyUI
<p align="center">
  <img src="./workflow.png" width=90%>
</p>

# img2img
 img2img can be done by send a image to the image imput in the sampler node,but the batch_size must be 1.

# StreamDiffusion_Sampler
Input Latent is not implemented for now.

  ### Lora stack
  You can set Lora stack by using LoRA Stacker from Efficiency Nodes.

## Support
Thank you for being awesome!