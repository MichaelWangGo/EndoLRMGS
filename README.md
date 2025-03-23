=======
# EndoLRMGS
=======
## Environment

cd ./EndoGaussian
wget https://github.com/CUHK-AIM-Group/EndoGaussian/tree/master/submodules
git submodule update --init --recursive
conda create -n EndoGaussian python=3.10
pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn

## Quick Start
- Example usage is as follows:
## Inference
  ```
endonerf/pulling:
python -m openlrm.launch infer.lrm --infer ./configs/infer-b-endonerf-pulling.yaml model_name=zxhezexin/openlrm-mix-base-1.1 image_input=/workspace/dataset/endolrm_dataset/endonerf/pulling/images export_mesh=false --gaussian_config /workspace/EndoLRM2/EndoGaussian/arguments/endonerf/pulling.py

endonerf/cutting:
python -m openlrm.launch infer.lrm --infer ./configs/infer-b-endonerf-cutting.yaml model_name=zxhezexin/openlrm-mix-base-1.1 image_input=/workspace/dataset/endolrm_dataset/endonerf/cutting_tissues_twice/images export_mesh=false --gaussian_config /workspace/EndoLRM2/EndoGaussian/arguments/endonerf/cutting_tissues_twice.py

stereomis:
python -m openlrm.launch infer.lrm --infer ./configs/infer-b-stereomis.yaml model_name=zxhezexin/openlrm-mix-base-1.1 image_input=/workspace/dataset/endolrm_dataset/stereomis/left_finalpass export_mesh=false --gaussian_config /workspace/EndoLRM2/EndoGaussian/arguments/stereomis/stereomis.py

scared:
python -m openlrm.launch infer.lrm --infer ./configs/infer-b-scared.yaml model_name=zxhezexin/openlrm-mix-base-1.1 image_input=/workspace/dataset/endolrm_dataset/scared/dataset_6/data/left_finalpass export_mesh=false --gaussian_config /workspace/EndoLRM2/EndoGaussian/arguments/scared/d6k4.py
  ```

## Training

  ```
endonerf/pulling:
accelerate launch --config_file ./configs/accelerate-train.yaml -m openlrm.launch train.lrm --config ./configs/train-sample.yaml --no-freeze_endo_gaussian --gaussian_config /workspace/EndoLRM2/EndoGaussian/arguments/endonerf/pulling.py

endonerf/cutting:
accelerate launch --config_file ./configs/accelerate-train.yaml -m openlrm.launch train.lrm --config ./configs/train-sample.yaml --no-freeze_endo_gaussian --gaussian_config /workspace/EndoLRM2/EndoGaussian/arguments/endonerf/cutting_tissues_twice.py

stereomis:
accelerate launch --config_file ./configs/accelerate-train.yaml -m openlrm.launch train.lrm --config ./configs/train-sample.yaml --no-freeze_endo_gaussian --gaussian_config /workspace/EndoLRM2/EndoGaussian/arguments/stereomis/stereomis.py

scared:
accelerate launch --config_file ./configs/accelerate-train.yaml -m openlrm.launch train.lrm --config ./configs/train-sample.yaml --no-freeze_endo_gaussian --gaussian_config /workspace/EndoLRM2/EndoGaussian/arguments/scared/d6k4.py
  ```

## Acknowledgement
Thank EndoGaussian https://github.com/CUHK-AIM-Group/EndoGaussian.git, LRM https://yiconghong.me/LRM/ and OpenLRM https://github.com/3DTopia/OpenLRM.git
