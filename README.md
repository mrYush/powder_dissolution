# How it work
## 1. Setup
```bash
poetry shell
poetry install
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
## 2. Run augmentation
```bash
cp src/augmentation/mk_augmentation_config.yml.example src/augmentation/mk_augmentation_config.yml
# Edit augmentation/mk_augmentation_config.yml
python stc/augmentation/mk_augmentation.py
```

## 3. Making annotation
We use https://www.makesense.ai/ to make annotation coresponding coco format [https://cocodataset.org/]

## 4. Train
```bash
cp src/train/mk_train_config.yml.example src/train/mk_train_config.yml
# Edit train/mk_train_config.yml be sure to set augmentation path which is generated in step 2
python stc/train/mk_train.py
```

## 5. Inference
```bash
cp src/inference/mk_predict_config.yml.example src/inference/mk_predict_config.yml
# Edit inference/mk_predict_config.yml be sure to set model path which is generated in step 4
python stc/inference/mk_predict.py
```

## 6. Calculate diffusion
```bash
cp src/diffusion/calc_diffusion_config.yml.example src/diffusion/calc_diffusion_config.yml
# Edit diffusion/calc_diffusion_config.yml be sure to set data_path which is generated in step 5 pickle file
python stc/diffusion/calc_diffusion.py
```
Find result.xlsx in result folder
