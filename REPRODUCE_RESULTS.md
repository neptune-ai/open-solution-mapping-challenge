## Prepare sources and data
* clone this repository
```bash
git clone https://github.com/minerva-ml/open-solution-mapping-challenge.git
```
* install requirements
```bash
pip3 install -r requirements.txt
```

**Note** 
You may need to install Cython by hand before pycocotools

* download the data from the [competition site](https://www.crowdai.org/challenges/mapping-challenge/dataset_files)
* register to [neptune.ml](http://bit.ly/2HtXtMH) *(if you wish to use it)* and remember your USERNAME :wink:
* login to [neptune.ml](http://bit.ly/2HtXtMH)
```bash
neptune account login
```
* open [neptune.ml](http://bit.ly/2HtXtMH) and create new project called: `Mapping-Challenge` with project key: `MC`
* go to your Neptune account and get api token. 
* set environment variables:

```bash
export NEPTUNE_API_TOKEN=your_api_token
export CONFIG_PATH=neptune.yaml
```

* change project name in `neptune.yaml`:
```yaml
project: user_name/project_name
```

## Prepare training data
* set paths in `neptune.yaml`
```yaml
data_dir:              /path/to/data
meta_dir:              /path/to/data
masks_overlayed_prefix: masks_overlayed
experiment_dir:        /path/to/work/dir
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
python main.py prepare_metadata --train_data --valid_data --test_data
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
python main.py evaluate_predict --pipeline_name unet
```

With Test time augmentation

```bash
python main.py evaluate_predict --pipeline_name unet_tta --chunk_size 1000
```

### Second level model (optional)

If you trained the second layer model go to the `pipeline_config.py` and change the `CATEGORY_LAYER` to 
what you chose during training. 
For example,

```python
CATEGORY_LAYERS = [1, 19]
```

```bash
python main.py evaluate_predict --pipeline_name unet_tta_scoring_model --chunk_size 1000
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
