# Installation
1. clone this repository: `git clone https://github.com/minerva-ml/open-solution-mapping-challenge.git`
2. install requirements: `pip3 install -r requirements.txt`
3. download the data from the competition site [dataset files](https://www.crowdai.org/challenges/mapping-challenge/dataset_files)
4. register to [Neptune](https://neptune.ml) *(if you wish to use it)* login via:

```bash
$ neptune login
```

open [Neptune](https://neptune.ml/ 'machine learning lab') and create new project called: `Mapping Challenge` with project key: `MC`*

upload the data to neptune (if you want to run computation in the cloud) via:
```bash
$ neptune data upload YOUR/DATA/FOLDER
```

5. prepare training data

   set paths in `neptune.yaml`

   ```yaml
	  data_dir:                   /path/to/data
	  meta_dir:                   /path/to/data
      masks_overlayed_prefix: masks_overlayed
	  experiment_dir:             /path/to/work/dir
   ```

   change erosion/dilation setup if in `neptune.yaml` you want to:
   Suggested setup is:

   ```yaml
    border_width: 0
	small_annotations_size: 14
	erode_selem_size: 0
	dilate_selem_size: 0
   ```

   prepare target masks and metadata for training:
   

    * local machine with neptune
    ```bash
    $ neptune login
    $ neptune experiment run \
    main.py -- prepare_masks  
    $ neptune experiment run \
    main.py -- prepare_metadata --train_data --valid_data --test_data 
    ```

    * cloud via neptune

    ```bash
    $ neptune login
    $ neptune experiment send --config neptune.yaml \
    --worker gcp-large \
    --environment pytorch-0.2.0-gpu-py3 \
    main.py -- prepare_masks
    $ neptune experiment send --config neptune.yaml \
    --worker gcp-large \
    --environment pytorch-0.2.0-gpu-py3 \
    main.py -- prepare_metadata --train_data --valid_data --test_data 
    ```

    * local pure python

    ```bash
    $ python main.py -- prepare_masks
    $ python main.py -- prepare_metadata --train_data --valid_data --test_data 
    ```

6. train model:

    * local machine with neptune
    ```bash
    $ neptune login
    $ neptune experiment run \
    main.py -- train --pipeline_name unet_weighted
    ```

    * cloud via neptune

    ```bash
    $ neptune login
    $ neptune experiment send --config neptune.yaml \
    --worker gcp-large \
    --environment pytorch-0.2.0-gpu-py3 \
    main.py -- train --pipeline_name unet_weighted
    ```

    * local pure python

    ```bash
    $ python main.py -- train --pipeline_name unet_weighted
    ```

7. evaluate model and predict on test data:
   Change values in the configuration file `neptune.yaml`.
   Suggested setup is:

   ```yaml
      tta_aggregation_method: gmean
      loader_mode: resize
      erode_selem_size: 0
      dilate_selem_size: 2
   ```

    * local machine with neptune
    ```bash
    $ neptune login
    $ neptune experiment run \
    main.py -- evaluate_predict --pipeline_name unet_tta --chunk_size 1000
    ```

    * cloud via neptune

    ```bash
    $ neptune login
    $ neptune experiment send --config neptune.yaml \
    --worker gcp-large \
    --environment pytorch-0.2.0-gpu-py3 \
    main.py -- evaluate_predict --pipeline_name unet_tta --chunk_size 1000
    ```

    * local pure python

    ```bash
    $ python main.py -- evaluate_predict --pipeline_name unet_tta --chunk_size 1000
    ```
