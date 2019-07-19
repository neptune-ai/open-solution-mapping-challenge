## Prepare sources and data
* clone this repository
```bash
git clone https://github.com/minerva-ml/open-solution-mapping-challenge.git
```
* install conda environment mapping

You can setup the project with default env variables and open `NEPTUNE_API_TOKEN` by running:

```bash
source make_project
```

I suggest at least reading the step-by-step instructions to know what is happening.

Install conda environment mapping

```bash
conda env create -f environment.yml
```

After it is installed you can activate/deactivate it by running:

```bash
conda activate mapping
```

```bash
conda deactivate
```

Register to the [neptune.ml](https://neptune.ml) _(if you wish to use it)_ even if you don't register you can still
see your experiment in Neptune. Just go to [shared/showroom project](https://ui.neptune.ml/o/shared/org/showroom/experiments) and find it.

Set environment variables `NEPTUNE_API_TOKEN` and `CONFIG_PATH`.

If you are using the default `neptune.yaml` config then run:
```bash
export export CONFIG_PATH=neptune.yaml
```

otherwise you can change to your config.

**Registered in Neptune**:

Set `NEPTUNE_API_TOKEN` variable with your personal token:

```bash
export NEPTUNE_API_TOKEN=your_account_token
```

Create new project in Neptune and go to your config file (`neptune.yaml`) and change `project` name:

```yaml
project: USER_NAME/PROJECT_NAME
``` 

**Not registered in Neptune**:

open token
```bash
export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJiNzA2YmM4Zi03NmY5LTRjMmUtOTM5ZC00YmEwMzZmOTMyZTQifQ==
```

## Prepare training data

* download the data from the [competition site](https://www.aicrowd.com/challenges/mapping-challenge#datasets)

We suggest setting up a following directory structure:

```
project
|--   README.md
|-- ...
|-- data
    |-- raw
         |-- train 
            |-- images 
            |-- annotation.json
         |-- val 
            |-- images 
            |-- annotation.json
         |-- test_images 
            |-- img1.jpg
            |-- img2.jpg
            |-- ...
    |-- meta
         |-- masks_overlayed_eroded_{}_dilated_{} # it is generated automatically
            |-- train 
                |-- distances 
                |-- masks 
                |-- sizes 
            |-- val 
                |-- distances 
                |-- masks 
                |-- sizes 
    |-- experiments
        |-- mapping_challenge_baseline # this is where your experiment files will be dumped
            |-- checkpoints # neural network checkpoints
            |-- transformers # serialized transformers after fitting
            |-- outputs # outputs of transformers if you specified save_output=True anywhere
            |-- prediction.json # prediction on valid
```

* set paths in `neptune.yaml` if you wish to use different project structure.
```yaml
  data_dir: data/raw
  meta_dir: data/meta
  masks_overlayed_prefix: masks_overlayed
  experiment_dir: data/experiments
```

* change erosion/dilation setup in `neptune.yaml` if you want to. Suggested setup
```yaml
border_width: 0
small_annotations_size: 14
erode_selem_size: 0
dilate_selem_size: 0
```

* prepare target masks and metadata for training
```bash
python main.py prepare_masks
python main.py prepare_metadata --train_data --valid_data
```

## Train model :rocket:

### Unet Network
This will train your neural network

```bash
python main.py train --pipeline_name unet_weighted
```

### Second level model (optional)
This will train a lightgbm to be able to get the best threshold.
Go to `pipeline_config.py` and change the number of thresholds to choose from for the building class.
19 means that your scoring model will learn which out of 19 threshold options (0.05...0.95) to choose for 
a particular image.

```python
CATEGORY_LAYERS = [1, 19]
```

```bash
python main.py train --pipeline_name scoring_model
```

## Evaluate model and predict on test data:

Change values in the configuration file `neptune.yaml`. 
Suggested setup:

```yaml
tta_aggregation_method: gmean
loader_mode: resize
erode_selem_size: 0
dilate_selem_size: 2
```

### Standard Unet evaluation

```bash
python main.py evaluate --pipeline_name unet
```

With Test time augmentation

```bash
python main.py evaluate --pipeline_name unet_tta --chunk_size 1000
```

### Second level model (optional)

If you trained the second layer model go to the `pipeline_config.py` and change the `CATEGORY_LAYER` to 
what you chose during training. 
For example,

```python
CATEGORY_LAYERS = [1, 19]
```

```bash
python main.py evaluate --pipeline_name unet_tta_scoring_model --chunk_size 1000
```


## Predict on new data

Put your images in some `inference_directory`.

Change values in the configuration file `neptune.yaml`. 
Suggested setup:

```yaml
tta_aggregation_method: gmean
loader_mode: resize
erode_selem_size: 0
dilate_selem_size: 2
```

Run prediction on this directory:

```bash
python main.py predict_on_dir \
--pipeline_name unet_tta_scoring_model \
--chunk_size 1000 \
--dir_path path/to/inference_directory \
--prediction_path path/to/predictions.json

```

## Enjoy results :trophy:
