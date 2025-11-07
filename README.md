# EndoLRMGS

### [arXiv Paper](https://arxiv.org/abs/2503.22437)
Abstract: Complete reconstruction of surgical scenes is crucial for robot-assisted surgery (RAS). Deep depth estimation is promising but existing works struggle with depth discontinuities, resulting in noisy predictions at object boundaries and do not achieve complete reconstruction omitting occluded surfaces. 
To address these issues we propose EndoLRMGS, that combines Large Reconstruction Modelling (LRM) and Gaussian Splatting (GS), for complete surgical scene reconstruction. GS reconstructs deformable tissues and LRM generates 3D models for surgical tools while position and scale are subsequently optimized by introducing orthogonal perspective joint projection optimization (OPjPO) to enhance accuracy.

## Architecture

![m3dris-architecture](media/pipeline.png)

## Prerequisites
- Python 3.10
- CUDA-compatible GPU

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/EndoLRMGS.git
cd EndoLRMGS

# Create and setup Python environment
conda create -n endolrmgs python=3.10
conda activate endolrmgs

# Install dependencies, we implement the work in docker with cuda 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd FMGaussianSplatting
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/simple-knn
cd ..
```

## Dataset Structure
Follow [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git) to prepare the required annotations and binary masks.

Example dataset structure:
```tree
endonerf/               # Root directory
├── pulling/
│   ├── Annotations/    # Surgical instruments color masks
│   ├── binary_mask_deva/ # Surgical instruments binary masks
│   ├── depth/         # Depth maps
│   └── images/        # Left rectified images
├── cutting_tissues_twice/
├── scared/
└── stereomis/
```

## Usage

### Inference
To access LRM pretrained model, please refer to https://huggingface.co/zxhezexin/openlrm-mix-base-1.1 and save the model in EndoLRMGS/checkpoint
For EndoNerf pulling dataset:
```bash
python -m openlrm.launch infer.lrm \
    --infer ./configs/infer-b-endonerf-pulling.yaml \
    --model_name zxhezexin/openlrm-mix-base-1.1 \
    --image_input /workspace/dataset/endolrm_dataset/endonerf/pulling/images \
    --export_mesh true \
    --gaussian_config ./FMGaussianSplatting/arguments/endonerf/pulling.py
```

For EndoNerf cutting dataset:
```bash
python -m openlrm.launch infer.lrm \
    --infer ./configs/infer-b-endonerf-cutting.yaml \
    --model_name zxhezexin/openlrm-mix-base-1.1 \
    --image_input /workspace/dataset/endolrm_dataset/endonerf/cutting_tissues_twice/images \
    --export_mesh true \
    --gaussian_config ./FMGaussianSplatting/arguments/endonerf/cutting.py
```

For Stereomis dataset:
```bash
python -m openlrm.launch infer.lrm \
    --infer ./configs/infer-b-stereomis.yaml \
    --model_name zxhezexin/openlrm-mix-base-1.1 \
    --image_input /workspace/datasets/endolrm_dataset/stereomis/new_p2_6/left_finalpass \
    --export_mesh true \
    --gaussian_config ./FMGaussianSplatting/arguments/stereomis/stereomis_2_6.py
```

For Scared dataset:
```bash
python -m openlrm.launch infer.lrm \
    --infer ./configs/infer-b-scared.yaml \
    --model_name zxhezexin/openlrm-mix-base-1.1 \
    --image_input /workspace/dataset/endolrm_dataset/scared/dataset_6/data/left_finalpass \
    --export_mesh true \
    --gaussian_config ./FMGaussianSplatting/arguments/scared/d1k1.py
```

### Training
To access LRM pretrained model "model.safetensors", please refer to https://huggingface.co/zxhezexin/openlrm-mix-small-1.1/tree/main and save the model in EndoLRMGS/checkpoint/mix-small
For EndoNerf pulling dataset:
```bash
accelerate launch --config_file ./configs/accelerate-train.yaml \
    -m openlrm.launch train.lrm \
    --config ./configs/train-sample.yaml \
    --no-freeze_gaussian \
    --gaussian_config ./FMGaussianSplatting/arguments/endonerf/pulling.py
```

For EndoNerf cutting dataset:
```bash
accelerate launch --config_file ./configs/accelerate-train.yaml \
    -m openlrm.launch train.lrm \
    --config ./configs/train-sample.yaml \
    --no-freeze_gaussian \
    --gaussian_config ./FMGaussianSplatting/arguments/endonerf/cutting.py
```

For Stereomis dataset:
```bash
accelerate launch --config_file ./configs/accelerate-train.yaml \
    -m openlrm.launch train.lrm \
    --config ./configs/train-sample.yaml \
    --no-freeze_gaussian \
    --gaussian_config ./FMGaussianSplatting/arguments/stereomis/stereomis_2_6.py
```

For Scared dataset:
```bash
accelerate launch --config_file ./configs/accelerate-train.yaml \
    -m openlrm.launch train.lrm \
    --config ./configs/train-sample.yaml \
    --no-freeze_gaussian \
    --gaussian_config ./FMGaussianSplatting/arguments/scared/d1k1.py
```

## Acknowledgements

This project builds upon the following works:
- [EndoGaussian](https://github.com/CUHK-AIM-Group/EndoGaussian.git)
- [LRM](https://yiconghong.me/LRM/)
- [OpenLRM](https://github.com/3DTopia/OpenLRM.git)
- [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git)
