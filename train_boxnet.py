import json
import os
import torch
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
# from universal_datamodule import UniversalDataModule
from utils.model_utils import (
    add_module_args,
    configure_optimizers,
)
from universal_checkpoint import UniversalCheckpoint
# from transformers import BertTokenizer, BertModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler, DDIMScheduler, PNDMScheduler
from torch.nn import functional as F
from PIL import Image
from PIL import ImageDraw
from tqdm.auto import tqdm
from universal_datamodule import DataModuleCustom
from typing import Callable, List, Optional, Union
import inspect
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from utils.utils import numpy_to_pil, save_colored_mask
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from boxnet_models import build_model, add_boxnet_args


blocks = [0,1,2,3]

def tokenize(tokenizer, prompts):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f for f in features if f is not None and isinstance(f, torch.Tensor)] # .float() .detach()
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f for k, f in features.items()} # .float()
        setattr(module, name, features)
    else: 
        setattr(module, name, features) # .float()

def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out

def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out

class StableDiffusion(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('OPPO Stable Diffusion Module')
        # parser.add_argument('--train_whole_model', action='store_true', default=False)
        parser.add_argument('--train_unet', action='store_true', default=False)
        parser.add_argument('--train_text_encoder', action='store_true', default=False)
        parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
        parser.add_argument('--timestep_range', type=int, default=[0, 1000], nargs="+")

        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.model_path, "text_encoder"))
        self.vae = AutoencoderKL.from_pretrained(
            args.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            args.model_path, subfolder="unet")
        self.test_scheduler = PNDMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )

        self.boxnet, self.boxnet_criterion, _ = build_model(args)

        self.save_hyperparameters(args)

        if not args.train_text_encoder:
            for _, param in self.text_encoder.named_parameters():
                param.requires_grad = False
        if not args.train_unet:
            for _, param in self.unet.named_parameters():
                param.requires_grad = False
        for _, param in self.vae.named_parameters():
            param.requires_grad = False
        
        save_hook = save_out_hook
        self.feature_blocks = []
        for idx, block in enumerate(self.unet.down_blocks):
            if idx in blocks:
                block.register_forward_hook(save_hook)
                self.feature_blocks.append(block) 
                
        for idx, block in enumerate(self.unet.up_blocks):
            if idx in blocks:
                block.register_forward_hook(save_hook)
                self.feature_blocks.append(block)  

    
    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = 40000
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        model_params = []
        if self.hparams.train_unet:
            model_params.append({'params': self.unet.parameters()})
        model_params.append({'params': self.boxnet.parameters()})
        return configure_optimizers(self, model_params=model_params)
    
    @torch.no_grad()
    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_input_ids = tokenize(self.tokenizer, prompt)

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
        )
        text_embeddings = text_embeddings[0]

        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]

            uncond_input_ids = tokenize(self.tokenizer, uncond_tokens)
            # pad_index = self.tokenizer.vocab['[PAD]']
            # uncond_attention_mask = uncond_input_ids.ne(pad_index).type(self.text_encoder.embeddings.word_embeddings.weight.dtype).to(device)

            uncond_embeddings = self.text_encoder(
                # uncond_input.input_ids.to(device),
                uncond_input_ids.to(device),
                # attention_mask=uncond_attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
    
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.test_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.test_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
        image = numpy_to_pil(image)
        return image
    
    def decode_latents_tensor(self, latents):
        self.vae.decoder.eval()
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        return image

    @torch.no_grad()
    def log_imgs(
        self,
        device,
        inputs,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        **kwargs,
    ):
        self.boxnet.eval()
        prompt = []
        cat_embeddings = []
        box_nums = []
        for data in inputs:
            prompt.append(data["prompt"])
            cat_input_id = tokenizer(
                data["phrases"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            tmp_embed = self.text_encoder(cat_input_id)[1]
            cat_embed = torch.zeros(args.num_queries, 768).to(device)
            cat_embed[:len(data["phrases"])] = tmp_embed
            cat_embeddings.append(cat_embed)
            box_nums.append(len(data["phrases"]))
        cat_embeddings = torch.stack(cat_embeddings)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self.encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        self.test_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.test_scheduler.timesteps

        shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device=device, dtype=text_embeddings.dtype)
        latents = latents * self.test_scheduler.init_noise_sigma

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        step_images = []
        for i, t in enumerate(tqdm(timesteps)):
            if t < args.timestep_range[0]:
                break
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.test_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            ################################################################
            # Extract activations
            activations = []
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

            activations = [activations[0][0], activations[1][0], activations[2][0], activations[3][0], activations[4], activations[5], activations[6], activations[7]]
            # activations = resize_and_concatenate(features, latents)

            assert all([isinstance(acts, torch.Tensor) for acts in activations])
            size = latents.shape[2:]
            resized_activations = []
            for acts in activations:
                acts = nn.functional.interpolate(
                    acts, size=size, mode="bilinear"
                )
                _, acts = acts.chunk(2)
                # acts = acts.transpose(1,3)
                resized_activations.append(acts)
            
            features =  torch.cat(resized_activations, dim=1)

            sqrt_one_minus_alpha_prod = (1 - self.test_scheduler.alphas_cumprod[t]).to(device) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(noise_pred.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            noise_level = sqrt_one_minus_alpha_prod * noise_pred
            # noise_level = noise_level.transpose(1,3)

            outputs = self.boxnet(features, noise_level, queries=cat_embeddings)  # .unflatten(0, (81, 64, 64)).transpose(3, 1)
            out_bbox = outputs['pred_boxes']
            boxes = box_cxcywh_to_xyxy(out_bbox)
            scale_fct = torch.tensor([width, height, width, height]).to(device)
            boxes = boxes * scale_fct
            out_images = []
            for m,box in enumerate(boxes):
                out_image = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(out_image)
                for n, b in enumerate(box[:box_nums[m]]):
                    draw.rectangle(((b[0], b[1]),(b[2], b[3])), fill=None, outline=n+1, width=5)
                out_images.append(out_image)
            step_images.append([(out_images[k], t.item()) for k in range(len(out_images)) ])
            ################################################################

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.test_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)

        return image, step_images

    @torch.no_grad()
    def get_bbox_input(self, batch):
        cat_embeddings = []
        masks = []
        for i, length in enumerate(batch["bbox_length"]): # batch loop
            cat_input_id = batch["cat_input_ids"][i][:length]
            tmp_embed = self.text_encoder(cat_input_id)[1]

            cat_embed = torch.zeros(args.num_queries, 768).to(tmp_embed.device)
            cat_embed[:length] = tmp_embed
            cat_embeddings.append(cat_embed)
            mask = torch.zeros((args.num_queries,)).to(tmp_embed.device)
            mask[:length] = 1
            masks.append(mask)
        return torch.stack(cat_embeddings), torch.stack(masks)

    def training_step(self, batch, batch_idx):
        self.boxnet.train()
        self.text_encoder.eval()

        with torch.no_grad():
            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * 0.18215
        
        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape).to(latents.device)
        noise = noise.to(dtype=self.unet.dtype)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            args.timestep_range[0], args.timestep_range[1], (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=self.unet.dtype)

        sqrt_alpha_prod = self.noise_scheduler.alphas_cumprod[timesteps].to(latents.device) ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(latents.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        noise_level = noisy_latents - (sqrt_alpha_prod * latents)

        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

        cat_embeddings, _ = self.get_bbox_input(batch)
        
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        activations = []
        # Extract activations
        for block in self.feature_blocks:
            activations.append(block.activations)
            block.activations = None
            
        activations = [activations[0][0], activations[1][0], activations[2][0], activations[3][0], activations[4], activations[5], activations[6], activations[7]]

        assert all([isinstance(acts, torch.Tensor) for acts in activations])
        size = latents.shape[2:]
        resized_activations = []
        for acts in activations:
            acts = nn.functional.interpolate(
                acts, size=size, mode="bilinear"
            )
            resized_activations.append(acts)
        
        features =  torch.cat(resized_activations, dim=1)

        outputs = self.boxnet(features, noise_level, queries=cat_embeddings)
        loss_dict = self.boxnet_criterion(outputs, batch['targets'])
        weight_dict = self.boxnet_criterion.weight_dict
        bbox_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        
        if args.train_unet or args.train_text_encoder:
            diffusion_loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            loss = diffusion_loss + bbox_loss * args.loss_proportion
            self.log("bbox_loss", bbox_loss.item(), on_epoch=False, prog_bar=True, logger=True)
            self.log("diffusion_loss", diffusion_loss.item(), on_epoch=False, prog_bar=True, logger=True)
        else:
            loss = bbox_loss
        
        self.log("bbox_loss", loss_dict["loss_bbox"].item(), on_epoch=False, prog_bar=True, logger=True)
        self.log("giou_loss", loss_dict["loss_giou"].item(), on_epoch=False, prog_bar=True, logger=True)
        self.log("train_loss", loss.item(), on_epoch=False, prog_bar=True, logger=True)
        self.log("lr", lr,  on_epoch=False, prog_bar=True, logger=True)

        if self.trainer.global_rank == 0:
            if (self.global_step+1) % args.save_steps == 0:
                print('saving model...')
                save_path = os.path.join(args.default_root_dir, f'hf_out_{self.trainer.current_epoch}_{self.global_step}')
                os.makedirs(save_path, exist_ok=True)
                model_path = os.path.join(save_path, f"boxnet.pt")
                torch.save(self.boxnet.state_dict(), model_path)
                #################################################
                if args.train_unet:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.model_path, unet=self.unet, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
                    )
                    pipeline.save_pretrained(os.path.join(
                        save_path, f'stable-diffusion'))
                #################################################
                print("testing model...")
                with torch.no_grad():
                    try:
                        with open(args.test_prompts, "r", encoding='utf-8') as f:
                            inputs = json.load(f)
                        assert inputs
                    except Exception:
                        print(f"No prompts read from file: {args.test_prompts}, skip test.")
                    else:
                        print(inputs)
                        images, step_images  = self.log_imgs(latents.device, inputs, num_images_per_prompt=1)
                        img_path = os.path.join(save_path, "test_images")
                        for i, img in enumerate(images):
                            res_path = os.path.join(img_path, "{}".format(i))
                            os.makedirs(res_path, exist_ok=True)
                            img.save(os.path.join(res_path, f"result.jpg"))
                            for cur_step_image in step_images:
                                step_image, t = cur_step_image[i]
                                step_image.save(os.path.join(res_path, f"step_{t}.png"))
                                save_colored_mask(os.path.join(res_path, f"color_step_{t}.png"), step_image)
                        print("Test images saved to: {}".format(img_path))

        return {"loss": loss}

    def on_train_epoch_end(self):
        if self.trainer.global_rank == 0:
            print('saving model...')
            save_path = os.path.join(args.default_root_dir, f'hf_out_{self.trainer.current_epoch}.pt')
            # save_config(self.bert_config, save_path)
            torch.save(self.lgp.state_dict(), save_path)

    def on_load_checkpoint(self, checkpoint) -> None:
        # 兼容低版本lightning，低版本lightning从ckpt起来时steps数会被重置为0
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = add_boxnet_args(args_parser)
    args_parser = DataModuleCustom.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = StableDiffusion.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    assert len(args.timestep_range) >= 2, "Must input timestep_range with (start, end)"

    model = StableDiffusion(args)
    tokenizer = model.tokenizer
    
    def collate_fn(examples):
        # print(examples)
        texts = []
        pixel_values = []
        label_values = []
        targets = []
        bbox_length = []
        cat_input_ids = []
        for example in examples:
            texts.append(example["instance_prompt"])
            pixel_values.append(example["instance_image"])
            label_values.append(example["instance_mask"])

            bbox_num = example["bbox_num"]
            if bbox_num > args.num_queries:
                cat_prompts = example["cat_prompts"][:args.num_queries]
                bbox = example["bbox"][:args.num_queries]
                expand_bbox = bbox
                labels = example["labels"][:args.num_queries]
                iscrowd = example["iscrowd"][:args.num_queries]
                bbox_num = args.num_queries
            else:
                bbox = example["bbox"]
                labels = example["labels"]
                iscrowd = example["iscrowd"]
                expand_bbox = torch.zeros((args.num_queries, 4))
                expand_bbox[:bbox_num] = bbox
                cat_prompts = [""] * args.num_queries
                cat_prompts[:bbox_num] = example["cat_prompts"]

            target = {
                'boxes': box_xyxy_to_cxcywh(bbox),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'iscrowd': torch.tensor(iscrowd),
                'orig_size': torch.tensor(example['orig_size']),
                'size': torch.tensor(example['size']),
            }
            targets.append(target)
            cat_input_id = tokenizer(
                cat_prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            cat_input_ids.append(cat_input_id)
            bbox_length.append(bbox_num)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        label_values = torch.stack(label_values)
        label_values = label_values.to(memory_format=torch.contiguous_format).int()
        cat_input_ids = torch.stack(cat_input_ids)

        input_ids = tokenizer(
                texts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
        
        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "targets": targets,
            "bbox_length": torch.tensor(bbox_length),
            "cat_input_ids": cat_input_ids
        }

        return batch

    datamoule = DataModuleCustom(
        args, tokenizer=tokenizer, collate_fn=collate_fn)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, datamoule, ckpt_path=args.load_ckpt_path)
