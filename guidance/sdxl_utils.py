from pathlib import Path

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import logging

# suppress partial model loading warning
logging.set_verbosity_error()
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision.utils import save_image

sys.path.append(".")
import lpips
import numpy as np
from diffusers.utils import load_image
from PIL import Image
from rembg import remove
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import pil_to_tensor

from utils.image_utils import psnr
from utils.loss_utils import l1_loss, l1_loss_weighted, l2_loss, l2_loss_weighted, lpips_loss, ssim

from .perpneg_utils import weighted_perpendicular_aggregator


def get_mask(x):
    x_copy = x.clone()
    if len(x_copy.shape) > 3:
        x_copy = x_copy[0]  # batch dim
    assert x_copy.shape[0] == 3
    input = ToPILImage()(x_copy)
    output = remove(input, only_mask=True)
    return torch.from_numpy(np.array(output)).to(x_copy.device)  # [H, W]


lpips_model = lpips.LPIPS(net="alex", verbose=False).cuda()

cur_sds_weight = 1
cur_diffusion_weight = 1

perpneg = True
as_latent = False
guidance_scale = 0
lambda_guidance = 1
save_guidance_path = None
vram_O = False
sd_version = "sdxl"
hf_key = None
device = "cuda"
fp16 = True
t_range = [0.02, 0.98]
refine = True
# text = "a tiger playing a guitar and holding a guitar in its paws and standing upright on a white background, furry art, Carlos Catasse, a digital painting, professional digital painting."


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


img_cache = {}


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16,
        vram_O,
        refine=False,
        sd_version="2.1",
        hf_key=None,
        t_range=[0.02, 0.98],
        is_turbo=True,
    ):
        super().__init__()
        self.device = device
        self.sd_version = sd_version

        print(f"[INFO] loading stable diffusion...")

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "./ckpts/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == "sdxl":
            if not is_turbo:
                model_key = "stabilityai/stable-diffusion-xl-base-1.0"
            else:
                model_key = "stabilityai/sdxl-turbo"
            self.is_turbo = is_turbo
        elif self.sd_version == "sdxl-ctrlnet":
            from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
            from transformers import DPTFeatureExtractor, DPTForDepthEstimation

            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
            self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0-small",
                variant="fp16",
                use_safetensors=True,
                torch_dtype=torch.float16,
            ).to("cuda")
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
            self.pipe_img2img = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                vae=vae,
                variant="fp16",
                use_safetensors=True,
                torch_dtype=torch.float16,
            ).to("cuda")
            self.pipe_img2img.enable_model_cpu_offload()
            self.strength = 0.5
            self.controlnet_conditioning_scale = 0.3
        else:
            raise ValueError(f"Stable-diffusion version {self.sd_version} not supported.")

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        if sd_version != "sdxl-ctrlnet":
            if refine:
                from diffusers import AutoencoderKL, AutoPipelineForImage2Image, DDIMScheduler
                from diffusers.pipelines import StableDiffusionXLImg2ImgPipeline

                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

                self.pipe_img2img: StableDiffusionXLImg2ImgPipeline = AutoPipelineForImage2Image.from_pretrained(
                    model_key, torch_dtype=torch.float16, variant="fp16", vae=vae
                )

                self.pipe_img2img.set_progress_bar_config(disable=True)
                self.pipe_img2img.to(device)
                self.tokenizer = self.pipe_img2img.tokenizer
                self.text_encoder = self.pipe_img2img.text_encoder
                self.unet = self.pipe_img2img.unet
                self.scheduler = self.pipe_img2img.scheduler
            else:
                self.scheduler = DDIMScheduler.from_pretrained(
                    model_key, subfolder="scheduler", torch_dtype=self.precision_t
                )

            self.num_train_timesteps = self.scheduler.config.num_train_timesteps
            self.min_step = int(self.num_train_timesteps * t_range[0])
            self.max_step = int(self.num_train_timesteps * t_range[1])
            self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f"[INFO] loaded stable diffusion!")

    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def train_step_diffusion(
        self,
        text,
        pred_rgb,
        guidance_scale=0.0,
        step=10,
        ref_image=None,
        pred_mask=True,
        psnr_threshold=35, #0,
        uid=None,
    ):
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # perform guidance (high scale from paper!)
        # noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        if uid is not None and uid in img_cache:
            image = img_cache[uid].to(ref_image)
        else:
            tmp_psnr = psnr(pred_rgb, ref_image).item()
            # print(tmp_psnr)
            if tmp_psnr > psnr_threshold:  # in case of accumulated demaging
                ref_image = pred_rgb
            # else:
            #     print("use gt image as ref...")

            if self.sd_version == "sdxl-ctrlnet":
                save_image(ref_image, "./tmp.png")
                image = load_image("./tmp.png").resize((1024, 1024))

                depth_image = self.get_depth_map(image)
                image = (
                    self.pipe_img2img(
                        text,
                        image=image,
                        control_image=depth_image,
                        strength=self.strength,
                        num_inference_steps=50,
                        controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                    )
                    .images[0]
                    .resize((576, 576))
                )
                image = pil_to_tensor(image)[None, ...].to("cuda") / 255
            elif self.sd_version == "sdxl":
                if self.is_turbo:
                    strength = 1 / step + 0.01
                    image = self.pipe_img2img.get_denoise(
                        prompt=text,
                        image=ref_image,
                        guidance_scale=guidance_scale,
                        num_inference_steps=int(step),
                        strength=strength,
                    )[0].float()
                else:
                    n_prompt = "text, blurry, fuzziness, bad face"
                    image = self.pipe_img2img.get_denoise(
                        prompt=text,
                        image=ref_image,
                        guidance_scale=guidance_scale,
                        negative_prompt=n_prompt,
                    ).float()
                # save_image(image, "./example.png")
                # save_image(ref_image, "./example_ref.png")

            if uid is not None:
                img_cache[uid] = image.cpu()

        image = torch.clamp(image, min=0.0, max=1.0)
        if pred_mask:
            mask = get_mask(image) / 255
            # mse_loss = l1_loss_weighted(pred_rgb, image, mask)  # mse loss
            image = (image * mask + torch.ones_like(image) * (1 - mask)).clamp(min=0.0, max=1.0)
            mse_loss = l1_loss(pred_rgb, image)
        else:
            mse_loss = l1_loss(pred_rgb, image)  # mse loss
        lpipsloss = lpips_loss(pred_rgb, image, lpips_model)
        # ssim_loss = ssim(pred_rgb, image)
        return mse_loss + 0.1 * lpipsloss

    def train_step_sds(self, text, pred_rgb, guidance_scale=0.0, grad_scale=1):
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # perform guidance (high scale from paper!)
        # noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        step = 10
        strength = 1 / step + 0.01
        _, noise_pred, noise, latents, timestep, image = self.pipe_img2img.get_noise_pred_v2(
            prompt=text,
            image=pred_rgb,
            guidance_scale=guidance_scale,
            num_inference_steps=int(step),
            strength=strength,
        )

        # from diffusers.utils import load_image
        # init_image = load_image(f"./pred.png") #.resize((512, 512))
        # image_ = self.pipe_img2img.__call__(
        #     prompt=text, image=init_image, guidance_scale=guidance_scale, num_inference_steps=int(step), strength=strength
        # ).images[0]
        # image[0].save("example.png")
        # image_.save("example_.png")

        # w(t), sigma_t^2
        w = 1 - self.alphas[int(timestep)]
        diff = w * (noise_pred - noise)  # latent
        grad = grad_scale * diff
        grad = torch.nan_to_num(grad)
        # latents.backward(gradient=grad, retain_graph=True)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
        return loss, diff.pow(2).mean().sum().item()

    @torch.no_grad()
    def produce_latents(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):

        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts)  # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(
            text_embeds,
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


guidance = nn.ModuleDict()
guidance["SD"] = StableDiffusion(device, fp16, vram_O, refine, sd_version, hf_key, t_range)


def get_sds_loss(text, image_tensor):
    if cur_sds_weight > 0:
        sd: StableDiffusion = guidance["SD"]
        loss_to_add, guidance_loss = sd.train_step_sds(
            text,
            image_tensor,
            guidance_scale=guidance_scale,
            grad_scale=lambda_guidance,
            # save_guidance_path=save_guidance_path,
        )
        return loss_to_add[0]


def get_diffusion_loss(image_tensor, ref_image=None, text=None, strength_step=10, uid=None):
    if cur_diffusion_weight > 0:
        sd: StableDiffusion = guidance["SD"]
        loss_to_add = sd.train_step_diffusion(
            text,
            image_tensor,
            guidance_scale=guidance_scale,
            step=strength_step,
            ref_image=ref_image,
            # save_guidance_path=save_guidance_path,
            uid=uid,
        )
        return loss_to_add


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument("--vram_O", action="store_true", help="optimization for low VRAM usage")
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
