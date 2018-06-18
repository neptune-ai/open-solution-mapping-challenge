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
* multistage training
* multigpu training


##### What didn't Work

##### We could work but we didnt' try it


#### Posprocessing

##### What Worked 

##### What didn't Work

##### We could work but we didnt' try it

We train a unet architecture with the encoder taken from resnet34 for the multiclass problem.

The implementation can be explored in unet_models proposed and implemented by Alexander Buslaev https://www.linkedin.com/in/al-buslaev/

We implemented loss weighing, where the closer the pixel is to the object the higher the weight is. It is based on the following paper https://arxiv.org/pdf/1505.04597.pdf

The parameters are specified in neptune.yaml
We added deeper resnet-based encoders( 34,101,152 flavours) (no experimental results yet)
We added loss weighing by object size to the cross entropy loss (no experimental results yet)
We added Dice loss to weigh it with size and distance weighted cross entropy to clean up masks (no experimental results yet)
We run inference on padded crops with replication of the last pixel (reflection pad didn't work to well)
Following postprocessing is applied to the output:

images are cropped to 300x300
CRF on masks and original images is performed to smoothen up the results. Because this approach is very slow we haven't used it in our current predictions. We will experiment later, having all other ideas in place.
instances of the same class are labeled
for each instance the dilation is performed with the same selem that was used for eroding the target masks in the first place
for each building area the mean probability is calculated

## Usage: Fast Track
1. clone this repository: `git clone https://github.com/neptune-ml/open-solution-mapping-challenge.git`
2. install requirements
3. register to [Neptune](https://neptune.ml/ 'machine learning lab') *(if you wish to use it)* login via:

```bash
$ neptune login
```

4. download/upload competition data and change data-related paths in the configuration file `neptune.yaml`
5. Prepare the target masks and data:

```bash
$ neptune experiment run main.py prepare_masks
$ neptune experiment run main.py prepare_metadata \
--train_data \
--valid_data \
--test_data
```

6. Put your competition API key in the configuration file
7. run experiment (for example via neptune):

```bash
$ neptune experiment run \
main.py train_evaluate_predict --pipeline_name unet --chunk_size 5000 --submit
```

7. check your leaderboard score!

## Usage: Detailed
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
