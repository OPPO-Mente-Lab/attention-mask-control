import json
import os
import torch
import argparse
import inspect
from typing import Callable, List, Optional, Union

import imgviz
import numpy as np
import torch.nn as nn
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from utils import numpy_to_pil
from boxnet_models import build_model, add_boxnet_args
from utils.box_ops import box_cxcywh_to_xyxy
from p2p import AttentionStore, show_cross_attention, EmptyControl


blocks = [0, 1, 2, 3]


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
        features = [f for f in features if f is not None and isinstance(
            f, torch.Tensor)]  # .float() .detach()
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f for k, f in features.items()}  # .float()
        setattr(module, name, features)
    else:
        setattr(module, name, features)  # .float()


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


def build_normal(u_x, u_y, d_x, d_y, step, device):
    x, y = torch.meshgrid(torch.linspace(0, 1, step),
                          torch.linspace(0, 1, step))
    x = x.to(device)
    y = y.to(device)
    out_prob = (1/2/torch.pi/d_x/d_y)*torch.exp(-1/2 *
                                                (torch.square((x-u_x)/d_x)+torch.square((y-u_y)/d_y)))
    return out_prob


def uniq_masks(all_masks, zero_masks=None, scale=1.0):
    uniq_masks = torch.stack(all_masks)
    # num = all_masks.shape[0]
    uniq_mask = torch.argmax(uniq_masks, dim=0)
    if zero_masks is None:
        all_masks = [((uniq_mask == i)*mask*scale).float().clamp(0, 1.0)
                     for i, mask in enumerate(all_masks)]
    else:
        all_masks = [((uniq_mask == i)*mask*scale).float().clamp(0, 1.0)
                     for i, mask in enumerate(zero_masks)]

    return all_masks


def build_masks(bboxes, size, mask_mode="gaussin_zero_one", focus_rate=1.5):
    all_masks = []
    zero_masks = []
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        mask = build_normal((y0+y1)/2, (x0+x1)/2, (y1-y0) /
                            4, (x1-x0)/4, size, bbox.device)
        zero_mask = torch.zeros_like(mask)
        zero_mask[int(y0 * size):min(int(y1 * size)+1, size),
                  int(x0 * size):min(int(x1 * size)+1, size)] = 1.0
        zero_masks.append(zero_mask)
        all_masks.append(mask)
    if mask_mode == 'zero_one':
        return zero_masks
    elif mask_mode == 'guassin':
        all_masks = uniq_masks(all_masks, scale=focus_rate)
        return all_masks
    elif mask_mode == 'gaussin_zero_one':
        all_masks = uniq_masks(all_masks, zero_masks, scale=focus_rate)
        return all_masks
    else:
        raise ValueError("Not supported mask_mode.")


def register_attention_control(model, controller, bboxes, entity_indexes, mask_control, mask_self=True, mask_mode='gaussin_zero_one', soft_mask_rate=0.5, focus_rate=1.5):
    def ca_forward(self, place_in_unet):

        def forward(hidden_states, context=None, mask=None):
            batch_size, sequence_length, channel = hidden_states.shape

            query = self.to_q(hidden_states)
            # context = context if context is not None else hidden_states
            is_cross = context is not None
            context = context if is_cross else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)

            dim = query.shape[-1]

            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1],
                            dtype=query.dtype, device=query.device),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )
            attention_probs = attention_scores.softmax(dim=-1)

            if mask_control:
                if is_cross:
                    size = int(np.sqrt(sequence_length))
                    all_masks = build_masks(
                        bboxes, size, mask_mode=mask_mode, focus_rate=focus_rate)
                    for pos, mask in zip(entity_indexes, all_masks):
                        start = pos[0]
                        end = pos[-1]
                        if mask.sum() <= 0:  # sequence_length *  0.004:
                            continue
                        mask = mask.reshape(
                            (sequence_length, -1)).to(hidden_states.device)
                        mask = mask.expand(-1, (end-start+1))
                        cond_attention_probs[:, :, start+1:end +
                                             2] = cond_attention_probs[:, :, start+1:end+2] * mask
                elif mask_self:
                    size = int(np.sqrt(sequence_length))
                    # must be 1/0
                    all_masks = build_masks(
                        bboxes, size, mask_mode=mask_mode, focus_rate=focus_rate)
                    for img_mask in all_masks:
                        if img_mask.sum() <= 0:
                            continue
                        img_mask = img_mask.reshape(sequence_length)
                        mask_index = img_mask.nonzero().squeeze(-1)
                        mask = torch.ones(sequence_length, sequence_length).to(
                            hidden_states.device)

                        mask[:, mask_index] = mask[:, mask_index] * \
                            img_mask.unsqueeze(-1)
                        cond_attention_probs = cond_attention_probs * mask + \
                            cond_attention_probs * (1-mask) * soft_mask_rate

            # # compute attention output
            attention_probs = controller(
                attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)

            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for name, net__ in net_.named_children():
                if 'fuser' not in name:
                    count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


class StableDiffusionTest():

    def __init__(self, model_path, device, boxnet_path=None):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(model_path, "text_encoder")).to(device)
        self.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet").to(device)
        self.test_scheduler = PNDMScheduler.from_pretrained(
            model_path, subfolder="scheduler")

        if boxnet_path is not None:
            args_parser = argparse.ArgumentParser()
            args_parser = add_boxnet_args(args_parser)
            args = args_parser.parse_args()
            args.no_class = True
            self.boxnet, _, _ = build_model(args)
            self.boxnet.load_state_dict(torch.load(boxnet_path))
            self.boxnet = self.boxnet.to(device)

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

    @torch.no_grad()
    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_input_ids = tokenize(self.tokenizer, prompt)

        # pad_index = self.tokenizer.vocab['[PAD]']
        # attention_mask = text_input_ids.ne(pad_index).type(self.text_encoder.embeddings.word_embeddings.weight.dtype).to(device)

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
        )
        text_embeddings = text_embeddings[0]
        # print("text_embeddings: ")
        # print(text_embeddings)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

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

            uncond_embeddings = self.text_encoder(
                # uncond_input.input_ids.to(device),
                uncond_input_ids.to(device),
                # attention_mask=uncond_attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(
            self.test_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.test_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

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
        controller=None,
        mask_control=False,
        mask_self=True,
        mask_mode='gaussin_zero_one',
        soft_mask_rate=0.5,
        focus_rate=1.5,
        **kwargs
    ):
        self.boxnet.eval()

        prompt = []
        cat_embeddings = []
        box_nums = []
        entities = []
        for data in inputs:
            prompt.append(data["prompt"])
            cat_input_id = self.tokenizer(
                data["phrases"],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            tmp_embed = self.text_encoder(cat_input_id)[1]
            cat_embed = torch.zeros(30, 768).to(device)
            cat_embed[:len(data["phrases"])] = tmp_embed
            cat_embeddings.append(cat_embed)
            box_nums.append(len(data["phrases"]))
            entities.append(data["entities"])
        cat_embeddings = torch.stack(cat_embeddings)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self.encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        self.test_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.test_scheduler.timesteps

        if latents is None:
            shape = (batch_size * num_images_per_prompt,
                     self.unet_stable.in_channels, height // 8, width // 8)
            latents = torch.randn(shape, generator=generator,
                                  device=device, dtype=text_embeddings.dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.test_scheduler.init_noise_sigma

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        all_boxes = []
        for i, t in enumerate(tqdm(timesteps)):
            # if i >= sample_step:
            #     break
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.test_scheduler.scale_model_input(
                latent_model_input, t)

            if controller is not None:
                register_attention_control(
                    self.unet, EmptyControl(), None, None, False)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t,
                                   encoder_hidden_states=text_embeddings).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

            ################################################################
            activations = []
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

            activations = [activations[0][0], activations[1][0], activations[2][0],
                           activations[3][0], activations[4], activations[5], activations[6], activations[7]]

            assert all([isinstance(acts, torch.Tensor)
                       for acts in activations])
            size = latents.shape[2:]
            resized_activations = []
            for acts in activations:
                acts = nn.functional.interpolate(
                    acts, size=size, mode="bilinear"
                )
                _, acts = acts.chunk(2)
                # acts = acts.transpose(1,3)
                resized_activations.append(acts)

            features = torch.cat(resized_activations, dim=1)

            sqrt_one_minus_alpha_prod = (
                1 - self.test_scheduler.alphas_cumprod[t]).to(device) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(noise_pred.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(
                    -1)
            noise_level = sqrt_one_minus_alpha_prod * noise_pred
            # noise_level = noise_level.transpose(1,3)

            # if i == sample_step:
            # .unflatten(0, (81, 64, 64)).transpose(3, 1)
            outputs = self.boxnet(features, noise_level,
                                  queries=cat_embeddings)
            out_bbox = outputs['pred_boxes']
            boxes = box_cxcywh_to_xyxy(out_bbox)
            new_boxes = boxes[0][:box_nums[0]]
            all_boxes.append(new_boxes)

            if controller is not None:
                register_attention_control(
                    self.unet, controller, boxes, entities, mask_control, mask_self=mask_self, mask_mode=mask_mode, soft_mask_rate=soft_mask_rate, focus_rate=focus_rate)
            noise_pred = self.unet(latent_model_input, t,
                                   encoder_hidden_states=text_embeddings).sample
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)
            if controller is not None:
                latents = controller.step_callback(latents)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.test_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs).prev_sample
            # break

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        if controller is not None:
            _, attn_img = show_cross_attention(
                prompt, self.tokenizer, controller, res=16, from_where=("up", "down"), save_img=False)

        return image, all_boxes, attn_img


def save_colored_mask(save_path, mask_pil):
    """保存调色板彩色图"""
    lbl_pil = mask_pil.convert('P')
    # lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    colormap = imgviz.label_colormap(80)
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def save_bbox_img(colored_res_path, bboxes, size=512, name="bbox.png"):
    scale_fct = torch.tensor([size, size, size, size]).to(device)
    bboxes = bboxes * scale_fct
    out_image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(out_image)
    for n, b in enumerate(bboxes):

        draw.rectangle(((b[0], b[1]), (b[2], b[3])),
                       fill=None, outline=n+1, width=5)
    save_colored_mask(os.path.join(colored_res_path, name), out_image)
    return out_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', default='test_prompts.json', type=str,
                        help="Prompt file to generate images.")
    parser.add_argument('--stable_model_path', required=True,
                        type=str, help="Original stable diffusion model path.")
    parser.add_argument('--boxnet_model_path', required=True,
                        type=str, help="BoxNet model path.")
    parser.add_argument('--output_dir', required=True, type=str,
                        help="Output dir for results.")
    parser.add_argument('--seed', default=1234, type=int,
                        help="Random seed.")
    parser.add_argument('--mask_mode', default='gaussin_zero_one', type=str, choices=['gaussin_zero_one', 'zero_one'],
                        help="mask mode.")
    parser.add_argument('--soft_mask_rate', default=0.5, type=float,
                        help="Soft mask rate for self mask.")
    parser.add_argument('--focus_rate', default=1.5, type=int,
                        help="Focus rate on area in-box")
    args = parser.parse_args()

    device = torch.device("cuda")
    with open(args.prompt_file, "r", encoding='utf-8') as f:
        inputs = json.load(f)
    model_path = args.stable_model_path
    boxnet_path = args.boxnet_model_path
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)

    amc_test = StableDiffusionTest(model_path, device, boxnet_path=boxnet_path)
    g_cpu = torch.Generator().manual_seed(args.seed)

    for i, cur_input in enumerate(inputs):
        print(inputs[i]['prompt'])
        cur_input['bboxes'] = None
        cur_path = os.path.join(save_path, "{}".format(i))
        os.makedirs(cur_path, exist_ok=True)
        controller = AttentionStore()
        images, all_step_bboxes, attn_img = amc_test.log_imgs(
            device, [cur_input], num_images_per_prompt=1, generator=g_cpu, controller=controller, mask_control=False, 
            mask_mode=args.mask_mode, soft_mask_rate=args.soft_mask_rate, focus_rate=args.focus_rate)

        images[0].save(os.path.join(cur_path, f"result.jpg"))
        for k, bboxes in enumerate(all_step_bboxes):
            save_bbox_img(cur_path, bboxes, name=f"bbox_{k}.png")
            # print(bboxes)
        attn_img.save(os.path.join(cur_path, f"attn.jpg"))
        controller = AttentionStore()
        images, all_step_bboxes, attn_img = amc_test.log_imgs(
            device, [cur_input], num_images_per_prompt=1, generator=g_cpu, controller=controller, mask_control=True, 
            mask_mode=args.mask_mode, soft_mask_rate=args.soft_mask_rate, focus_rate=args.focus_rate)

        images[0].save(os.path.join(cur_path, f"masked_result.jpg"))
        for k, bboxes in enumerate(all_step_bboxes):
            save_bbox_img(cur_path, bboxes, name=f"masked_bbox_{k}.png")
            # print(bboxes)
        attn_img.save(os.path.join(cur_path, f"masked_attn.jpg"))
