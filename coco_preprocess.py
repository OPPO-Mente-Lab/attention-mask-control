from pycocotools.coco import COCO
from PIL import Image
import os
import argparse
import cv2
import random
import numpy as np
from tqdm import tqdm
import webdataset as wds


def load_captions(coco, imgId):
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)
    captions = []
    for ann in anns:
        captions.append(ann['caption'])
    return captions


def load_bbox(coco, imgId, names, filename):
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)
    
    ann_dict = {
        'image_id': imgId,
        'filename': filename,
        "cat_ids": [],
        "cat_names": [],
        "is_crowd": [],
        "bboxes": []
    }
    for ann in anns:
        cat = coco.loadCats([ann['category_id']])[0]
        cat_name = cat['name']
        cat_idx = names.index(cat_name) + 1   # re-map
        ann_dict['cat_ids'].append(cat_idx)
        ann_dict['cat_names'].append(cat_name)
        ann_dict['is_crowd'].append(ann["iscrowd"])
        x,y,w,h = ann["bbox"]
        ann_dict['bboxes'].append([x,y,x+w,y+h])
    return ann_dict



def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_num = args.save_num

    coco_caption = COCO(args.coco_caption_file)
    coco_bbox = COCO(args.coco_instance_file)
    imgIds = coco_bbox.getImgIds()
    catIds = coco_bbox.getCatIds()       # 类别ID列表
    cats = coco_bbox.loadCats(catIds)   # 获取类别信息->dict
    print("catIds len: {}, imgIds len: {}".format(len(catIds), len(imgIds)))
    names = [cat['name'] for cat in cats]  # 类名称
    print(names)

    zero_num = 0

    save_pairs = []
    for idx, imgId in tqdm(enumerate(imgIds), ncols=100):
        image = coco_bbox.loadImgs([imgId])[0]
        filename = image['file_name']
        image_path = os.path.join(args.coco_image_path, filename)
        if os.path.isfile(args.coco_mask_path):
            mask_path = os.path.join(args.coco_mask_path, filename.replace('.jpg', '.png'))
        else:
            mask_path = ''

        ann_dict = load_bbox(coco_bbox, imgId, names, filename)
        if not ann_dict["bboxes"]:
            zero_num += 1
        captions = load_captions(coco_caption, imgId)
        for caption in captions:
            save_pairs.append((image_path, mask_path, caption, ann_dict))
        if idx % 10000 == 0:
            print('Loaded {}/{} images.'.format(idx, len(imgIds)))
    print(zero_num)
    print(len(imgIds))
    random.shuffle(save_pairs)

    save_shard_num = 0
    output_file = os.path.join(output_dir, "{}.tar".format(str(save_shard_num).zfill(5)))
    dst = wds.TarWriter(output_file)
    save_last = True
    for i, (image_path, mask_path, caption, ann_dict) in tqdm(enumerate(save_pairs)):
        image = Image.open(image_path)
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = Image.new("L", image.size, 0)
        dst.write({"__key__":"{}".format(str(i).zfill(5)), "jpg":image, 'png': mask, "txt":caption.strip(), "json": ann_dict})
        if i % save_num == save_num-1:
            print(f"Saved tar file: {output_file}")
            dst.close()
            if i < len(save_pairs)-1:
                save_shard_num += 1
                output_file = os.path.join(output_dir, "{}.tar".format(str(save_shard_num).zfill(5)))
                dst = wds.TarWriter(output_file)
            else:
                save_last = False
    if save_last:
        print(f"Saved tar file: {output_file}")
        dst.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_image_path', required=True, type=str,
                        help="COCO image path.")
    parser.add_argument('--coco_mask_path', default='',
                        type=str, help="COCO mask path, if exists.")
    parser.add_argument('--coco_caption_file', required=True,
                        type=str, help="COCO caption file, default name is 'captions_train2014.json'.")
    parser.add_argument('--coco_instance_file', required=True, type=str,
                        help="COCO caption file, default name is 'instances_train2014.json'.")
    parser.add_argument('--output_dir', required=True, type=str,
                        help="Where to save converted tar file.")
    parser.add_argument('--save_num', default=100, type=int,
                        help="How many text-image pairs to save in one tar file.")
    args = parser.parse_args()

    main(args)



