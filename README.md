# How to Use the Code Repository

Below are detailed instructions for each step of using the code repository for analysis of powder dissolution images. 
These instructions are intended to be used as a template for your own project.
Adjust the instructions as necessary to fit the actual configuration parameters and file paths used in your project.

## 1. Setup

Before you begin, set up your Python environment and install all required dependencies using the following commands:

```bash
poetry shell
poetry install
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 2. Run Data Augmentation

To augment your image data, follow these steps. 
Prepare your images in a directory and specify the path to this directory in the configuration file `mk_augmentation_config.yml`.
Images should be in the `.jpg` format.

```bash
cp src/augmentation/mk_augmentation_config.yml.example src/augmentation/mk_augmentation_config.yml
# Edit the mk_augmentation_config.yml to specify the paths to your input images and desired output directory for augmented images.
python stc/augmentation/mk_augmentation.py
```

After running the script, check the output directory specified in `mk_augmentation_config.yml` to verify that your images have been augmented correctly.

## 3. Making Annotations

Annotations are made using the [MakeSense.ai](https://www.makesense.ai/) tool. The images you plan to annotate should be the output from Step 2. Make sure to export the annotations in the COCO format. Detailed guidance can be found at [COCO Dataset](https://cocodataset.org/).

## 4. Train the Model

To train the model with your dataset, you should have annotations in the COCO format (in a JSON file) and corresponding images in a directory.

```bash
cp src/train/mk_train_config.yml.example src/train/mk_train_config.yml
# Edit the mk_train_config.yml to set the 'augmentation_path' to the directory containing the augmented images and annotations from Step 2 and Step 3.
python stc/train/mk_train.py
```

This script will use the augmented images and corresponding annotations to train the model. 
The trained model will be saved to the path specified in the configuration file.

## 5. Inference

After training the model, you can use it to make predictions on new images.
Put the images you want to predict in a directory and specify the path to this directory in the configuration file `mk_predict_config.yml`.

```bash
cp src/inference/mk_predict_config.yml.example src/inference/mk_predict_config.yml
# In the mk_predict_config.yml, set the 'model_path' to where your trained model is saved from Step 4.
python stc/inference/mk_predict.py
```

The inference script will process the images specified in the `mk_predict_config.yml` file and produce predictions based on the trained model. These predictions are typically saved as a pickle file or similar format for further analysis.

## 6. Calculate Diffusion Coefficients

Finally, to calculate the diffusion coefficients:

```bash
cp src/diffusion/calc_diffusion_config.yml.example src/diffusion/calc_diffusion_config.yml
# Edit the calc_diffusion_config.yml to point 'data_path' to the pickle file generated in Step 5.
python stc/diffusion/calc_diffusion.py
```

Ensure that the input path is set to the location of your inference results from Step 5. The script will calculate the diffusion coefficients and save the results in an Excel file named `result.xlsx` in the `result` folder.

By following these steps and correctly configuring the paths in each configuration file, you should be able to process your image data from augmentation to diffusion coefficient calculation successfully.
