
import io
import random
import braceexpand

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


USED_KEYS = {"jpg": "instance_images", "txt": "instance_prompt_ids", "json": "bboxes", "png": "instance_masks"}

cat_list = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


prompt_list = [
    "{}"
    # "a {}",
    # "a photo of {}",
]


def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)


def expand_text(text):
    pat = random.choice(prompt_list)
    return pat.format(text)


def custom_decoder(key, data):
    "自定义decoder，处理原图和mask，原图处理方式等同于pil或者pilrgb"
    import PIL.Image
    if key.endswith("png"):
        # return None
        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            result = img.convert("L")
            return result
    elif key.endswith("jpg"):
        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            result = img.convert("RGB")
            return result
    else:
        return None


def verify_keys(samples, required_keys, hr_size, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """
    for sample in samples:
        try:
            for key in required_keys:
                assert key in sample, f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}"
            # assert sample['jpg'].size == sample['png'].size, f"Sample {sample['__key__']} jpg must have same shape as png."
            # assert len(sample['json']["bboxes"]) > 0, "No bbox found in {}".format(sample['json']["filename"])
            # 测试阶段，注释掉过滤规则
            # if (hr_size < 0 or (sample['json']['original_width'] >= hr_size and sample['json']['original_height'] >= hr_size)) and (2 <= len(sample['txt']) <= 32):
            if len(sample['json']["bboxes"]) > 0:
                yield {key:sample[key] for key in required_keys}
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break
key_verifier = wds.filters.pipelinefilter(verify_keys)

class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            tokenizer=None,
            extra_keys=[],
            hr_size=-1,
            size= 512,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True,
            center_crop=False,
            merge_cat=False,
            shuffle_cat=False
    ):
        super().__init__()
        keys = list(USED_KEYS.keys()) + extra_keys
        # self.key_map = {key: i for i, key in enumerate(keys)}
        self.resampling = resample
        self.hr_size = hr_size
        self.size = size
        self.merge_cat = merge_cat
        self.shuffle_cat = shuffle_cat
        self.cat2idx = {k: i for i,k in enumerate(cat_list)}
        
        self.tokenizer = tokenizer

        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        else:
            self.append(wds.SimpleShardList(urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        self.append(wds.tarfile_to_samples(handler=handler))

        self.append(wds.decode(custom_decoder, handler=handler))

        self.append(key_verifier(required_keys=keys, hr_size=hr_size, handler=handler))
        # Apply preprocessing
        self.append(wds.map(self.preproc))

    def transform(self, image, bbox, mask=None):
        # Resize
        resize_img = transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR)
        image = resize_img(image)
        if mask is not None:
            resize_mask = transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST)
            mask = resize_mask(mask)

        # Transform to tensor
        image = TF.to_tensor(image)

        if mask is not None:
            mask = torch.from_numpy(np.array(mask, np.uint8, copy=True)).unsqueeze(-1)
            mask = mask.permute((2, 0, 1)).contiguous()

        # norm image
        norm = transforms.Normalize([0.5], [0.5])
        image = norm(image)
        if mask is not None:
            resize_mask_new = transforms.Resize(self.size//8, interpolation=transforms.InterpolationMode.NEAREST)
            mask = resize_mask_new(mask)
        return image, bbox, mask


    def preproc(self, sample):
        # 过滤太小的bbox
        example = {}
        instance_image = sample["jpg"]
        w, h = instance_image.size
        example["orig_size"] = [int(h), int(w)]
        example["size"] = [int(h), int(w)]
        bbox = torch.tensor(sample["json"]["bboxes"])
        bbox = bbox / torch.tensor([w,h,w,h]) # 归一化
        assert bbox.max() <= 1, "Wrong bbox"
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_image"], bbox, example["instance_mask"] = self.transform(instance_image, bbox, sample['png'])
        example["instance_prompt"] = sample["txt"]

        if not self.merge_cat:
            example["bbox"] = bbox
            example["cat_prompts"] = [expand_text(name) for name in sample["json"]["cat_names"]]
            example["bbox_num"] = len(sample["json"]["bboxes"])
            example["labels"] = [self.cat2idx[name] for name in sample["json"]["cat_names"]]
            example["iscrowd"] = sample["json"]["is_crowd"]

            areas = (example["bbox"][:, 2] - example["bbox"][:, 0]) * (example["bbox"][:, 3] - example["bbox"][:, 1])
            cat_labels = {}
            include_idx = []
            for i ,area in enumerate(areas):
                if area < 0.00244140625:
                    continue
                else:
                    include_idx.append(i)
                label = example["labels"][i]
                if label in cat_labels:
                    cat_labels[label].append(i)
                else:
                    cat_labels[label] = [i]
            if len(include_idx) > 0:
                include_idx = torch.tensor(include_idx).long()
                for label, idxes in cat_labels.items():
                    new_area = areas[idxes]
                    new_bbox = example["bbox"][idxes]
                    sort_idxes = torch.argsort(new_area, descending=True)
                    example["bbox"][idxes] = new_bbox[sort_idxes]
                example["bbox"] = example["bbox"][include_idx]
                example["bbox_num"] = len(include_idx)
                example["labels"] = [l for i, l in enumerate(example["labels"]) if i in include_idx]
                example["iscrowd"] = [c for i, c in enumerate(example["iscrowd"]) if i in include_idx]
                example["cat_prompts"] = [prompt for i, prompt in enumerate(example["cat_prompts"]) if i in include_idx]
                if example["bbox_num"] > 30:
                    areas = areas[include_idx]
                    area_sort_idxes = torch.argsort(areas, descending=True)[:30]
                    example["bbox"] = example["bbox"][area_sort_idxes]
                    example["bbox_num"] = 30
                    example["labels"] = [l for i, l in enumerate(example["labels"]) if i in area_sort_idxes]
                    example["iscrowd"] = [c for i, c in enumerate(example["iscrowd"]) if i in area_sort_idxes]
                    example["cat_prompts"] = [prompt for i, prompt in enumerate(example["cat_prompts"]) if i in area_sort_idxes]
        else:
            merged_cat_names = list(set(sample["json"]["cat_names"]))
            example["cat_prompts"] = [expand_text(name) for name in merged_cat_names]
            example["bbox_num"] = len(merged_cat_names)
            example["labels"] = [self.cat2idx[name] for name in merged_cat_names]
        
        return example


if __name__ == '__main__':
    from transformers import CLIPTokenizer
    # from lightning.pytorch import seed_everything
    # seed_everything(23)
    url = "/public_data/wrc/data/coco_bbox/train_tars/{}.tar"
    available_shards = list(range(0, 2))
    urls = [url.format(str(shard).zfill(5)) for shard in available_shards]
    ds = ImageEmbeddingDataset(
                urls,
                shuffle_shards=True,
                resample=False,
                hr_size=512,
                handler=wds.handlers.warn_and_continue
            )
    # ds.with_epoch(1)
    ds.with_epoch(200)
    for i, data in enumerate(iter(ds)):
        print(i)
        break

    tokenizer = CLIPTokenizer.from_pretrained("/public_data/wrc/models/stable-diffusion-v1-5", subfolder="tokenizer")
    def collate_fn(examples):
        texts = []
        pixel_values = []
        cat_input_ids = []
        bboxes = []
        bbox_length = []
        for example in examples:
            texts.append(example["instance_prompt"])
            pixel_values.append(example["instance_image"])

            bbox_num = example["bbox_num"]

            if bbox_num > 30:
                cat_prompts = example["cat_prompts"][:30]
                bbox = example["bbox"][:30]
                bbox_num = 30
            else:
                bbox = torch.zeros((30, 4))
                bbox[:bbox_num] = example["bbox"]
                cat_prompts = [""] * 30
                cat_prompts[:bbox_num] = example["cat_prompts"]
            cat_input_id = tokenizer(
                cat_prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            cat_input_ids.append(cat_input_id)
            bboxes.append(bbox)
            bbox_length.append(bbox_num)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer(
                texts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
        bboxes = torch.stack(bboxes)
        cat_input_ids = torch.stack(cat_input_ids)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "bbox_length": torch.tensor(bbox_length),
            "bboxes": bboxes,
            "cat_input_ids": cat_input_ids
        }

        return batch

    from prefetch_generator import BackgroundGenerator
    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
    
    loader = DataLoaderX(
            ds,
            num_workers=0,
            batch_size=64,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn
        )
    for i, batch in enumerate(loader):
        print(batch['pixel_values'].shape)
        # break
        print(i)

