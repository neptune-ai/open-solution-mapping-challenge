#!/bin/bash

# Provided to facilitate running the 'Predict on new data' step in the REPRODUCE file.
export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJiNzA2YmM4Zi03NmY5LTRjMmUtOTM5ZC00YmEwMzZmOTMyZTQifQ==
export CONFIG_PATH=neptune.yaml

python main.py predict-on-dir \
--pipeline_name unet_tta_scoring_model \
--chunk_size 1000 \
--dir_path images/ \
--prediction_path images/predictions.json
