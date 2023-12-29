import os
import sys
from .streamdiffusion.pipeline import StreamDiffusion
from .streamdiffusion.wrapper import StreamDiffusionWrapper
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict
import torch
import gc
# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.abspath(os.path.join(my_dir, '..'))
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Construct the path to the font file
font_path = os.path.join(my_dir, 'arial.ttf')

# Append comfy_dir to sys.path & import files
sys.path.append(comfy_dir)
import folder_paths


import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.latent_formats
import comfy.model_management

# Append my_dir to sys.path & import files
sys.path.append(my_dir)

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np

MAX_RESOLUTION=8192

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_async_loop():
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class StreamDiffusion_Loader:
    @classmethod
    def INPUT_TYPES(s):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "ckpt_name": (["Baked ckpt"]+folder_paths.get_filename_list("checkpoints"), ),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "lcm_lora": (loras,),
                "acceleration": (["none", "xfomers", "sfast", "tensorrt"],),
                
                "use_tiny_vae": ("BOOLEAN", { "default": True }),
                "use_lcm_lora": ("BOOLEAN", { "default": True }),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", ),
            },
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
        
    FUNCTION = "efficientloader" 
    CATEGORY = "StreamDiffusion/Loader"
 
    def efficientloader(self,ckpt_name,vae_name,lcm_lora,acceleration,use_tiny_vae,use_lcm_lora,lora_stack=None):

        device = comfy.model_management.get_torch_device()
        device_name = comfy.model_management.get_torch_device_name(device)
        vae_dtype=comfy.model_management.vae_dtype()

        if ckpt_name =='Baked ckpt':
            ckpt_path="KBlueLeaf/kohaku-v2.1"
        else:
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        if vae_name=='Baked VAE':
            vae_id = None
        else:
            vae_id=vae_name

        if lcm_lora=='None':
            lcm_lora_id=None
        else:
            lcm_lora_id= folder_paths.get_full_path("loras", lcm_lora)

        lora_dict =None
        if lora_stack is not None:
            lora_dict={}
            for lora_name, lora_scale, strength_clip in lora_stack:
                full_lora_name=folder_paths.get_full_path("loras", lora_name)
                lora_dict[full_lora_name]=lora_scale

        t_index_list=[32,40,45]

        stream = StreamDiffusionWrapper(
            model_id_or_path=ckpt_path,
            lora_dict=lora_dict,
            t_index_list=t_index_list,
            frame_buffer_size=1,
            width=512,
            height=512,
            warmup=10,
            acceleration=acceleration,
            use_tiny_vae =use_tiny_vae,
            device=device,
            use_lcm_lora = use_lcm_lora,
            output_type = 'pt',
            dtype = torch.float16,
            lcm_lora_id=lcm_lora_id,
            vae_id =vae_id,
        )

        return (stream,)
        

class StreamDiffusion_Sampler:
    @classmethod
    def INPUT_TYPES(s):
         return {
                "required":{
                    "model": ("MODEL",),
                    "positive": ("STRING", {"default": "CLIP_POSITIVE","multiline": True}),
                    "negative": ("STRING", {"default": "CLIP_NEGATIVE", "multiline": True}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 100.0}),
                    "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),

                    "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                    "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),#The frame buffer size for denoising batch, by default 1.

                    "index_list": ("STRING", {"default": "32,40,45","multiline": False}),

                    "cfg_type": (["none", "full", "self", "initialize"],),
                  
                    "add_noise": ("BOOLEAN", { "default": True }),
                    "use_denoising_batch": ("BOOLEAN", { "default": True }),
                    "enable_similar_image_filter": ("BOOLEAN", { "default": False }),
                    "use_safety_checker": ("BOOLEAN", { "default": False }),

                    },
                "optional": {
                    "similar_image_filter_threshold": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 100.0,"step": 0.01}),
                    "similar_image_filter_max_skip_frame": ("INT", {"default": 10, "min": 0, "max": 100}),
                    "latent": ("LATENT",),
                    "image": ("IMAGE",),
                    "lora_stack": ("LORA_STACK", ),
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
        
    FUNCTION = "sample" 
    CATEGORY = "StreamDiffusion/Sampler"
 
    @torch.no_grad()
    def sample(self,model,positive,negative,seed,steps,cfg,delta,width,height,batch_size,index_list,cfg_type,add_noise,use_denoising_batch,enable_similar_image_filter=False,use_safety_checker=False,similar_image_filter_threshold= 0.98,similar_image_filter_max_skip_frame=10,latent=None,image=None,lora_stack=None):
        device = comfy.model_management.get_torch_device()
        device_name = comfy.model_management.get_torch_device_name(device)
        vae_dtype=comfy.model_management.vae_dtype()

        # latent_image = latent["samples"]
        # latent_image = latent_image.to(model.device)
        # batch_size,channel,latent_height,latent_width,=latent_image.shape
        # width=latent_width*8
        # height =latent_height*8

        t_index_list=[32,40,45]
        t_index_list =[int(i) for i in index_list.split(',')] 

        if image ==None:
            mode = "txt2img"
        else:
            mode = "img2img"
            image = image.movedim(-1,1)

        if cfg <= 1.0:
            cfg_type = "none"

        if batch_size>1 and mode=="txt2img":
            use_denoising_batch=False
      
        
        # stream.set_sampler_param(t_index_list=t_index_list,
        #     width=width,
        #     height=height,
        #     do_add_noise= add_noise=='enable',
        #     frame_buffer_size=frame_buffer_size,
        #     use_denoising_batch=use_denoising_batch,
        #     cfg_type=cfg_type,)

        model.prepare(
            positive,
            negative,
            steps,
            cfg,
            delta,
            t_index_list,
            add_noise,
            enable_similar_image_filter,
            similar_image_filter_threshold,
            similar_image_filter_max_skip_frame,
            use_denoising_batch,
            cfg_type,
            seed,
            batch_size,
            use_safety_checker,
        )
        # latent_image = self.predict_x0_batch(
        #     torch.randn((stream.batch_size, 4, stream.latent_height, stream.latent_width)).to(
        #         device=stream.device, dtype=stream.dtype
        #     )
        # )

        # if batch_size==1:
        #     for _ in range(stream.batch_size - 1):
        #         stream()

        output = model.sample(image).permute(0, 2, 3, 1)
       
        return (output,)


NODE_CLASS_MAPPINGS = {
    "StreamDiffusion_Loader": StreamDiffusion_Loader,
    "StreamDiffusion_Sampler":StreamDiffusion_Sampler,
    
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusion_Loader": "StreamDiffusion_Loader",
    "StreamDiffusion_Sampler":"StreamDiffusion_Sampler",
    
}