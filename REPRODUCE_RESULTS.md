## Prepare sources and data
* clone this repository
```bash
git clone https://github.com/minerva-ml/open-solution-mapping-challenge.git
```
* install requirements
```bash
pip3 install -r requirements.txt
```
* download the data from the [competition site](https://www.crowdai.org/challenges/mapping-challenge/dataset_files)
* register to [neptune.ml](https://neptune.ml) *(if you wish to use it)* and remember your USERNAME :wink:
* login to [neptune.ml](https://neptune.ml)
```bash
neptune account login
```
* open [neptune.ml](https://neptune.ml) and create new project called: `Mapping-Challenge` with project key: `MC`
* upload the data to [neptune.ml](https://neptune.ml) (if you want to run computations in the cloud) via:
```bash
neptune data upload -r --project USERNAME/Mapping-Challenge YOUR/DATA/FOLDER
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
    * local machine with neptune
        ```bash
        neptune login
        neptune run main.py -- prepare_masks
        neptune run main.py -- prepare_metadata --train_data --valid_data --test_data 
        ```
    * cloud via neptune
        ```bash
        neptune login
        neptune send --worker gcp-large --environment pytorch-0.2.0-gpu-py3 main.py -- prepare_masks
        neptune send --worker gcp-large --environment pytorch-0.2.0-gpu-py3 main.py -- prepare_metadata --train_data --valid_data --test_data
        ```
    * local pure python
        ```bash
        python main.py -- prepare_masks
        python main.py -- prepare_metadata --train_data --valid_data --test_data
        ```

## Train model :rocket:
* local machine with neptune
    ```bash
    neptune login
    neptune run main.py -- train --pipeline_name unet_weighted
    ```
* cloud via neptune
    ```bash
    neptune login
    neptune send --worker gcp-large --environment pytorch-0.2.0-gpu-py3 main.py -- train --pipeline_name unet_weighted
    ```
* local pure python
    ```bash
    python main.py -- train --pipeline_name unet_weighted
    ```

## Evaluate model and predict on test data:
Change values in the configuration file `neptune.yaml`. Suggested setup:
```yaml
tta_aggregation_method: gmean
loader_mode: resize
erode_selem_size: 0
dilate_selem_size: 2
```
* local machine with neptune
    ```bash
    neptune login
    neptune run main.py -- evaluate_predict --pipeline_name unet_tta --chunk_size 1000
    ```
* cloud via neptune
    ```bash
    neptune login
    neptune send --worker gcp-large --environment pytorch-0.2.0-gpu-py3 main.py -- evaluate_predict --pipeline_name unet_tta --chunk_size 1000
    ```
* local pure python
    ```bash
    python main.py -- evaluate_predict --pipeline_name unet_tta --chunk_size 1000
    ```

## Enjoy results :trophy:
