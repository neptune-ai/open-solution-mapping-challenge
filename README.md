# open-solution-mapping-challenge

[![Join the chat at https://gitter.im/minerva-ml/open-solution-mapping-challenge](https://badges.gitter.im/minerva-ml/open-solution-mapping-challenge.svg)](https://gitter.im/minerva-ml/open-solution-mapping-challenge?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Open solution to the [CrowdAI Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge)

## Goal
Deliver open source, ready-to-use and extendable solution to this competition. This solution should - by itself - establish solid benchmark, as well as provide good base for your custom ideas and experiments.

## Results

Our approach got `0.943` Average Precision and `0.954` Average Recall on stage 1 data. You can check our experiments [here](https://app.neptune.ml/neptune-ml/Mapping-Challange)
Some examples (no cherry-picking I promise):

<img src="https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/0f88863b18904b23d4301611ddf2b532aff8de96/example_output.png"></img>

I have to say that the results exceded my expectations. The output from the network is so good that not a lot of morphological shenanigans is needed. Happy days:)

## Solution write-up

We implemented the following pipeline:

<img src="https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/e1bf6300fa119db2fec6622a603c63655ff5d770/unet_pipeline.png"></img>

#### Preprocessing

##### What Worked 

* Overlay binary masks for each image is produced [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/preparation.py)
* Distances to the 2 closest objects are calculated creating the distance map that is used for weighing [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/preparation.py)
* Size masks for each image is produced [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/preparation.py)
* Dropped small masks on the edges [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/preparation.py#L141-L142)
* We load training and validation data in batches:
using `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` makes it easy and clean (see loaders.py )[code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/loaders.py)
only some basic augmentations (due to speed constraints) from the imgaug package are be applied to images (see augmentations.py )
* Image is resized before feeding it to the network. Surprisingly this worked better than cropping [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/loaders.py#L246-L263) [config](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/neptune.yaml#L47)

##### What didn't Work

* Ground truth masks are prepared by first eroding them per mask creating non overlapping masks and only after that the distances are calculated [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/preparation.py)
* Dilated small objectcs to increase the signal (no experimental results yet) [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/preparation.py)
* Network is fed with random crops [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/loaders.py#L225-L243) [config](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/neptune.yaml#L47)

##### What could have worked but we haven't tried it

* Ground truth masks for overlapping contours (DSB-2018 winners approach)

#### Network

##### What Worked 

* Unet with Resnet101 as encoder. The approach is explained https://arxiv.org/abs/1806.00844 [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/unet_models.py#L315-L403) [config](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/neptune.yaml#L63)

##### What didn't Work

* Unet build from scracth with Resnet34 and Resnet152 as encoder. Worked to a certain degree but failed to produce the very best results. [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/steps/pytorch/architectures/unet.py) 
* Network architecture based on dilated convolutions described here https://arxiv.org/pdf/1709.00179.pdf

##### What could have worked but we haven't tried it
* Unet with contextual blocks explained here https://openreview.net/pdf?id=S1F-dpjjM


#### Loss

##### What Worked 

* distance weighted cross entropy explained here https://arxiv.org/pdf/1505.04597.pdf [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/models.py#L227-L371) [config](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/neptune.yaml#L79-L80)
* using linear combination of soft dice and distance weighted cross entropy [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/models.py#L227-L371) [config](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/neptune.yaml#L65-L67)
* adding size weighted component to the weighted cross entropy that would penalize misclassification on pixels belonging to small objects [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/models.py#L227-L371)

Inputs to the  distance and size weighted cross entropy look like this:

<img src="https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/1578b08c464dd3829bb3437e4534ce6d1eafc632/loss_inputs.png"></img>


##### What didn't Work

##### What could have worked but we haven't tried it


#### Training

##### What Worked 
* use pretrained models
* use multistage training
1. train on a 50000 subset of the dataset with `lr=0.0001` and `dice_weight=0.5`
2. train on a full dataset with `lr=0.0001` and `dice_weight=0.5`
3. train with smaller `lr=0.00001` and `dice_weight=0.5`
4. increase dice weight to `dice_weight=5.0` to make results smoother
* multigpu training
* use very simple augmentations

The entire configuration can be tweaked from the config file [config](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/neptune.yaml)

##### What didn't Work

##### We could work but we didnt' try it
* set different learning rates to different layers
* use cyclic optimizers
* use warm start optimizers

#### Postprocessing

##### What Worked 

* test time augmentations rotations + flips and geometric mean [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/loaders.py#L338-L497) [config](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/pipeline_config.py#L119-L125)
* simple morphological operations. In the beginning we used erosion followed by labeling and per label dilation with structure elements chosed by CV but as the models got better erosion was removed and very small dilation was the only one showing improvements [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/postprocessing.py)
* scoring objects. In the beginning we simply used score `1.0` for every object which was a huge mistake. 
Changing that to average probability over the object region improved results. What improved scores even more was weighing those probabilities with the object size. [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/postprocessing.py#L173-L181)
* second level model

##### What didn't Work
* test time augmentations colors [config](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/pipeline_config.py#L122)
* inference on reflection-padded images was not a way to go. What worked better (but not for the very best models) was replication padding where border pixel value was replicated for all the padded regions. [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/loaders.py#L313)
* Conditional Random Fields. To be honest it was so slow that we didn't check it for the best models [code](https://github.com/minerva-ml/open-solution-mapping-challenge/blob/dev/src/postprocessing.py#L128-L170)

##### What could have worked but we haven't tried it
* Ensembling
* Recurrent Neural networks for cleanup


## Installation
1. clone this repository: `git clone https://github.com/minerva-ml/open-solution-mapping-challenge.git`
2. install requirements: `pip3 install -r requirements.txt`
3. download the data from the competition site [dataset files](https://www.crowdai.org/challenges/mapping-challenge/dataset_files)
4. register to [Neptune](https://neptune.ml/ 'machine learning lab') *(if you wish to use it)* login via:

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


    * local machine with neptune
    ```bash
    $ neptune login
    $ neptune experiment run \
    main.py -- prepare_metadata --train_data --valid_data --test_data 
    ```

    * cloud via neptune

    ```bash
    $ neptune login
    $ neptune experiment send --config neptune.yaml \
    --worker gcp-large \
    --environment pytorch-0.2.0-gpu-py3 \
    main.py -- prepare_metadata --train_data --valid_data --test_data 
    ```

    * local pure python

    ```bash
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
    $ python main.py train --pipeline_name unet_weighted
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
    $ python main.py evaluate_predict --pipeline_name unet_tta --chunk_size 1000
    ```

## User support
There are several ways to seek help:
1. crowdai [discussion](https://www.crowdai.org/challenges/mapping-challenge/topics) is our primary way of communication.
1. You can submit an [issue](https://github.com/minerva-ml/open-solution-mapping-challenge/issues) directly in this repo.

## Contributing
1. Check [CONTRIBUTING](CONTRIBUTING.md) for more information.
1. Check [issues](https://github.com/minerva-ml/open-mapping-challenge/issues) and [project](https://github.com/minerva-ml/open-solution-mapping-challenge/projects/1) to check if there is something you would like to contribute to.
