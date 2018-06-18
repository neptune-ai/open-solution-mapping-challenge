# open-solution-mapping-challenge

[![Join the chat at https://gitter.im/minerva-ml/open-solution-mapping-challenge](https://badges.gitter.im/minerva-ml/open-solution-mapping-challenge.svg)](https://gitter.im/minerva-ml/open-solution-mapping-challenge?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Open solution to the [CrowdAI Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge)

## Goal
Deliver open source, ready-to-use and extendable solution to this competition. This solution should - by itself - establish solid benchmark, as well as provide good base for your custom ideas and experiments.

## Results

Our approach got `0.943` Average Precision and `0.954` Average Recall on stage 1 data.
Some examples (no cherry-picking I promise):

<img src="https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/0f88863b18904b23d4301611ddf2b532aff8de96/example_output.png"></img>

I have to say that the results exceded my expectations. The output from the network is so good that not a lot of morphological is needed. Happy days:)

## Solution write-up

We implemented the following pipeline:

<img src="https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/e1bf6300fa119db2fec6622a603c63655ff5d770/unet_pipeline.png"></img>

#### Preprocessing

##### What Worked 

* Distances to the 2 closest objects are calculated creating the distance map that is used for weighing
* Dropped small masks on the edges
* We load training and validation data in batches:
using `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` makes it easy and clean (see loaders.py )
only some basic augmentations (due to speed constraints) from the imgaug package are be applied to images (see augmentations.py )
* Image is resized before feeding it to the network. Surprisingly this worked better than cropping

##### What didn't Work

* Ground truth masks are prepared by first eroding them per mask creating non overlapping masks and only after that the distances are calculated
* Dilated small objectcs to increase the signal (no experimental results yet)
* Network is fed with random crops

##### What could work but we didnt' try it

* Ground truth masks for overlapping contours (DSB-2018 winners approach)

#### Network

##### What Worked 

* Unet with Resnet101 as encoder. The approach is explained https://arxiv.org/abs/1806.00844 

##### What didn't Work

* Unet build from scracth with Resnet34 and Resnet152 as encoder. Worked to a certain degree but failed to produce the very best results.
* Network architecture based on dilated convolutions described here https://arxiv.org/pdf/1709.00179.pdf

##### What could work but we didnt' try it
* Unet with contextual blocks explained here https://openreview.net/pdf?id=S1F-dpjjM


#### Loss

##### What Worked 

* distance weighted cross entropy explained here https://arxiv.org/pdf/1505.04597.pdf
* using linear combination of soft dice and distance weighted cross entropy
* adding size weighted component to the weighted cross entropy that would penalize misclassification on pixels belonging to small objects

##### What didn't Work

##### What could work but we didnt' try it


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

##### What didn't Work

##### We could work but we didnt' try it
* set different learning rates to different layers
* use cyclic optimizers
* use warm start optimizers

#### Postprocessing

##### What Worked 

* test time augmentations rotations + flips and geometric mean
* simple morphological operations. In the beginning we used erosion followed by labeling and per label dilation with structure elements chosed by CV but as the models got better erosion was removed and very small dilation was the only one showing improvements
* scoring objects. In the beginning we simply used score `1.0` for every object which was a huge mistake. 
Changing that to average probability over the object region improved results. What improved scores even more was weighing those probabilities with the object size. 
* second level model

##### What didn't Work
* test time augmentations colors 
* inference on reflection-padded images was not a way to go. What worked better (but not for the very best models) was replication padding where border pixel value was replicated for all the padded regions.
* Conditional Random Fields. To be honest it was so slow that we didn't check it for the best models

##### We could work but we didnt' try it
* Ensembling
* Recurrent Neural networks for cleanup

## Installation
1. clone this repository: `git clone https://github.com/minerva-ml/open-solution-talking-data.git`
2. install [PyTorch](http://pytorch.org/) and `torchvision`
3. install requirements: `pip3 install -r requirements.txt`
4. register to [Neptune](https://neptune.ml/ 'machine learning lab') *(if you wish to use it)* login via:

```bash
$ neptune login
```

5. open [Neptune](https://neptune.ml/ 'machine learning lab') and create new project called: `Mapping Challenge` with project key: `MC`
6. download the data from the competition site
7. upload the data to neptune (if you want to run computation in the cloud) via:
```bash
$ neptune data upload YOUR/DATA/FOLDER
```

8. change paths in the `neptune.yaml` .

```yaml
  data_dir:               /path/to/data
  meta_dir:               /path/to/data
  masks_overlayed_dir:    /path/to/masks_overlayed
  experiment_dir:         /path/to/work/dir
```

9. run experiment:

    * local machine with neptune
    ```bash
    $ neptune login
    $ neptune experiment run \
    main.py -- train_evaluate_predict --pipeline_name unet --chunk_size 5000
    ```

    * cloud via neptune

    ```bash
    $ neptune login
    $ neptune experiment send --config neptune.yaml \
    --worker gcp-large \
    --environment pytorch-0.2.0-gpu-py3 \
    main.py -- train_evaluate_predict --pipeline_name solution_1 --chunk_size 5000
    ```

    * local pure python

    ```bash
    $ python main.py train_evaluate_predict --pipeline_name unet --chunk_size 5000
    ```

## User support
There are several ways to seek help:
1. crowdai [discussion](https://www.crowdai.org/challenges/mapping-challenge/topics) is our primary way of communication.
1. You can submit an [issue](https://github.com/minerva-ml/open-solution-mapping-challenge/issues) directly in this repo.

## Contributing
1. Check [CONTRIBUTING](CONTRIBUTING.md) for more information.
1. Check [issues](https://github.com/minerva-ml/open-mapping-challenge/issues) and [project](https://github.com/minerva-ml/open-solution-mapping-challenge/projects/1) to check if there is something you would like to contribute to.
