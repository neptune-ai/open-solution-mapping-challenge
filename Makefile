# create env
conda env create -f environment.yml # --force # -vvv

# create directories
mkdir -p data/raw data/meta data/experiments

# set default env variable for NEPTUNE_API_TOKEN and CONFIG_PATH
export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJiNzA2YmM4Zi03NmY5LTRjMmUtOTM5ZC00YmEwMzZmOTMyZTQifQ==
export CONFIG_PATH=neptune.yaml
