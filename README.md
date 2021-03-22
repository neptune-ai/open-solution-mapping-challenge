# Open Solution to the Mapping Challenge Competition

[![Gitter](https://badges.gitter.im/minerva-ml/open-solution-mapping-challenge.svg)](https://gitter.im/minerva-ml/open-solution-mapping-challenge?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/LICENSE)

# Note
**Unfortunately, we can no longer provide support for this repo. Hopefully, it should still work, but if it doesn't, we cannot really help.** 

## More competitions :sparkler:
Check collection of [public projects :gift:](https://ui.neptune.ai/-/explore), where you can find multiple Kaggle competitions with code, experiments and outputs.

## Poster :earth_africa:
Poster that summarizes our project is [available here](https://gist.github.com/kamil-kaczmarek/b3b939797fb39752c45fdadfedba3ed9/raw/7fa365392997e9eae91c911c1837b45bfca45687/EP_poster.pdf).

## Intro
Open solution to the [CrowdAI Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge) competition.
1. Check **live preview of our work** on public projects page: [Mapping Challenge](https://ui.neptune.ai/neptune-ai/Mapping-Challenge) [:chart_with_upwards_trend:](https://ui.neptune.ai/neptune-ai/Mapping-Challenge).
1. Source code and [issues](https://github.com/neptune-ai/open-solution-mapping-challenge/issues) are publicly available.

## Results
`0.943` **Average Precision** :rocket:

`0.954` **Average Recall** :rocket:

<img src="https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/0f88863b18904b23d4301611ddf2b532aff8de96/example_output.png"></img>

_No cherry-picking here, I promise :wink:. The results exceded our expectations. The output from the network is so good that not a lot of morphological shenanigans is needed. Happy days:)_

Average Precision and Average Recall were calculated on [stage 1 data](https://www.crowdai.org/challenges/mapping-challenge/dataset_files) using [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools). Check this [blog post](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173) for average precision explanation.

## Disclaimer
In this open source solution you will find references to the neptune.ai. It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ai is not necessary to proceed with this solution. You may run it as plain Python script :wink:.

## Reproduce it!
Check [REPRODUCE_RESULTS](REPRODUCE_RESULTS.md)

# Solution write-up
## Pipeline diagram

<img src="https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/e1bf6300fa119db2fec6622a603c63655ff5d770/unet_pipeline.png"></img>

## Preprocessing
### :heavy_check_mark: What Worked
* Overlay binary masks for each image is produced ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/preparation.py) :computer:).
* Distances to the two closest objects are calculated creating the distance map that is used for weighing ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/preparation.py) :computer:).
* Size masks for each image is produced ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/preparation.py) :computer:).
* Dropped small masks on the edges ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/preparation.py#L141-L142) :computer:).
* We load training and validation data in batches: using [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) makes it easy and clean ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/loaders.py) :computer:).
* Only some basic augmentations (due to speed constraints) from the [imgaug package](https://github.com/aleju/imgaug) are applied to images ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/augmentation.py) :computer:).
* Image is resized before feeding it to the network. Surprisingly this worked better than cropping ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/loaders.py#L246-L263) :computer: and [config](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/neptune.yaml#L47) :bookmark_tabs:).

### :heavy_multiplication_x: What didn't Work
* Ground truth masks are prepared by first eroding them per mask creating non overlapping masks and only after that the distances are calculated ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/preparation.py) :computer:).
* Dilated small objects to increase the signal ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/preparation.py) :computer:).
* Network is fed with random crops ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/loaders.py#L225-L243) :computer: and [config](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/neptune.yaml#L47) :bookmark_tabs:).

### :thinking: What could have worked but we haven't tried it
* Ground truth masks for overlapping contours ([DSB-2018 winners](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741) approach).

## Network
### :heavy_check_mark: What Worked
* Unet with Resnet34, Resnet101 and Resnet152 as an encoder where Resnet101 gave us the best results. This approach is explained in the [TernausNetV2](https://arxiv.org/abs/1806.00844) paper (our [code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/unet_models.py#L315-L403) :computer: and [config](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/neptune.yaml#L63) :bookmark_tabs:). Also take a look at our parametrizable [implementation of the U-Net](https://github.com/neptune-ai/steppy-toolkit/blob/master/toolkit/pytorch_transformers/architectures/unet.py#L9).

### :heavy_multiplication_x: What didn't Work
* Network architecture based on dilated convolutions described in [this paper](https://arxiv.org/abs/1709.00179).

### :thinking: What could have worked but we haven't tried it
* Unet with contextual blocks explained in [this paper](https://openreview.net/pdf?id=S1F-dpjjM).

## Loss function
### :heavy_check_mark: What Worked
* Distance weighted cross entropy explained in the famous [U-Net paper](https://arxiv.org/pdf/1505.04597.pdf) (our [code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/models.py#L227-L371) :computer: and [config](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/neptune.yaml#L79-L80) :bookmark_tabs:).
* Using linear combination of soft dice and distance weighted cross entropy ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/models.py#L227-L371) :computer: and [config](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/neptune.yaml#L65-L67) :bookmark_tabs:).
* Adding component weighted by building size (smaller buildings has greater weight) to the weighted cross entropy that penalizes misclassification on pixels belonging to the small objects ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/models.py#L227-L371) :computer:).

### Weights visualization
For both weights: the darker the color the higher value.
* distance weights: high values corresponds to pixels between buildings.
* size weights: high values denotes small buildings (the smaller the building the darker the color). Note that no-building is fixed to black.

<img src="https://gist.githubusercontent.com/jakubczakon/cac72983726a970690ba7c33708e100b/raw/1578b08c464dd3829bb3437e4534ce6d1eafc632/loss_inputs.png"></img>

## Training
### :heavy_check_mark: What Worked
* Use pretrained models!
* Our multistage training procedure:
    1. train on a 50000 examples subset of the dataset with `lr=0.0001` and `dice_weight=0.5`
    1. train on a full dataset with `lr=0.0001` and `dice_weight=0.5`
    1. train with smaller `lr=0.00001` and `dice_weight=0.5`
    1. increase dice weight to `dice_weight=5.0` to make results smoother
* Multi-GPU training
* Use very simple augmentations

The entire configuration can be tweaked from the [config file](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/neptune.yaml) :bookmark_tabs:.

### :thinking: What could have worked but we haven't tried it
* Set different learning rates to different layers.
* Use cyclic optimizers.
* Use warm start optimizers.

## Postprocessing
### :heavy_check_mark: What Worked
* Test time augmentation (tta). Make predictions on image rotations (90-180-270 degrees) and flips (up-down, left-right) and take geometric mean on the predictions ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/loaders.py#L338-L497) :computer: and [config](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/pipeline_config.py#L119-L125) :bookmark_tabs:).
* Simple morphological operations. At the beginning we used erosion followed by labeling and per label dilation with structure elements chosed by cross-validation. As the models got better, erosion was removed and very small dilation was the only one showing improvements ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/postprocessing.py) :computer:).
* Scoring objects. In the beginning we simply used score `1.0` for every object which was a huge mistake. Changing that to average probability over the object region improved results. What improved scores even more was weighing those probabilities with the object size ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/postprocessing.py#L173-L181) :computer:).
* Second level model. We tried Light-GBM and Random Forest trained on U-Net outputs and features calculated during postprocessing.

### :heavy_multiplication_x: What didn't Work
* Test time augmentations by using colors ([config](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/pipeline_config.py#L122) :bookmark_tabs:).
* Inference on reflection-padded images was not a way to go. What worked better (but not for the very best models) was replication padding where border pixel value was replicated for all the padded regions ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/loaders.py#L313) :computer:).
* Conditional Random Fields. It was so slow that we didn't check it for the best models ([code](https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/postprocessing.py#L128-L170) :computer:).

### :thinking: What could have worked but we haven't tried it
* Ensembling
* Recurrent neural networks for postprocessing (instead of our current approach)

# Model Weights

Model weights for the winning solution are available [here](https://ui.neptune.ai/o/neptune-ai/org/Mapping-Challenge/e/MC-1057/artifacts)

You can use those weights and run the pipeline as explained in [REPRODUCE_RESULTS](REPRODUCE_RESULTS.md).

# User support
There are several ways to seek help:
1. crowdai [discussion](https://www.crowdai.org/challenges/mapping-challenge/topics).
1. You can submit an [issue](https://github.com/neptune-ai/open-solution-mapping-challenge/issues) directly in this repo.
1. Join us on [Gitter](https://gitter.im/minerva-ml/open-solution-mapping-challenge?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge).

# Contributing
1. Check [CONTRIBUTING](CONTRIBUTING.md) for more information.
1. Check [issues](https://github.com/neptune-ai/open-solution-mapping-challenge/issues) to check if there is something you would like to contribute to.
