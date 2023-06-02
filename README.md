### Code for paper: "Compositional Text-to-Image Synthesis with Attention Map Control of Diffusion Models"
[[Paper](https://arxiv.org/abs/2305.13921)]

# Requirements
A suitable [conda](https://conda.io/) environment named `amc` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate amc
```

# Data Prepearing
First, please download the coco dataset from [here](https://cocodataset.org/ "here"). We use COCO2014 in the paper.
Then, you can process your data with this script:
```shell
python coco_preprocess.py \
    --coco_image_path /YOUR/COCO/PATH/train2014 \
    --coco_caption_file /YOUR/COCO/PATH/annotations/captions_train2014.json \
    --coco_instance_file /YOUR/COCO/PATH/annotations/instances_train2014.json \
    --output_dir /YOUR/DATA/PATH
```

# Training
Before training, you need to change configs in `train_boxnet.sh`
- ROOT_DIR: where to save all the results.
- webdataset_base_urls: /YOUR/DATA/PATH/{xxx-xxx}.tar
- model_path: stable diffusion v1-5 checkpoint

You can train the BoxNet through this script:
```shell
sh train_boxnet.sh
```
# Text-to-Image Synthesis
With a trained BoxNet, you can start the Text-to-Image Synthesis with:
```shell
python test_pipeline_onestage.py \
	--stable_model_path /stable-diffusion-v1-5/checkpoint
	--boxnet_model_path /TRAINED/BOXNET/CKPT
	--output_dir /YOUR/SAVE/DIR
```
all the test prompt is saved in file `test_prompts.json`.

# TODOs

- [x] Release data preparation code
- [x] Release inference code
- [x] Release training code
- [ ] Release demo
- [ ] Release checkpoint


# Acknowledgements 
This implementation is based on the repo from the [diffusers](https://github.com/huggingface/diffusers) library. 
[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/finetune_taiyi_stable_diffusion) codebase.
[DETR](https://github.com/facebookresearch/detr) codebase.
