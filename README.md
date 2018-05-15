# open-solution-mapping-challenge

[![Join the chat at https://gitter.im/minerva-ml/open-solution-mapping-challenge](https://badges.gitter.im/minerva-ml/open-solution-mapping-challenge.svg)](https://gitter.im/minerva-ml/open-solution-mapping-challenge?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Open solution to the [CrowdAI Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge)

## Goal
Deliver open source, ready-to-use and extendable solution to this competition. This solution should - by itself - establish solid benchmark, as well as provide good base for your custom ideas and experiments.

## Usage: Fast Track
1. clone this repository: `git clone https://github.com/neptune-ml/open-solution-mapping-challenge.git`
2. install requirements
3. register to [Neptune](https://neptune.ml/ 'machine learning lab') *(if you wish to use it)* login via:

```bash
$ neptune login
```

4. download/upload competition data and change data-related paths in the configuration file `neptune.yaml`
5. Prepare the data:

```bash
$ neptune experiment run main.py -- prepare_metadata \
--train_data \
--valid_data \
--test_data
```

6. Put your competition API key in the configuration file
7. run experiment (for example via neptune):

```bash
$ neptune login
$ neptune experiment run \
main.py -- train_evaluate_predict --pipeline_name unet --chunk_size 5000
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
