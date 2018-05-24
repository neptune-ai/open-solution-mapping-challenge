import yaml
import subprocess
import os

import click

MISSING_TRANSFORMERS = ['prediction_crop',
                        'prediction_renamed',
                        'mask_resize',
                        'category_mapper',
                        'mask_erosion',
                        'labeler',
                        'mask_dilation',
                        'score_builder',
                        'output']


@click.group()
def main():
    pass


@main.command()
@click.option('-e', '--experiment_dir', help='experiment that you want to run evaluation on', required=True)
@click.option('-t', '--temp_inference_dir', help='temporary directory', required=True)
def run(temp_inference_dir, experiment_dir):
    transformer_dir = os.path.join(temp_inference_dir, 'transformers')
    checkpoints_dir = os.path.join(temp_inference_dir, 'checkpoints')

    cmd = 'cp -rf {} {}'.format(experiment_dir, temp_inference_dir)
    subprocess.call(cmd, shell=True)

    cmd = 'cp {}/unet/best.torch {}/unet'.format(checkpoints_dir, transformer_dir)
    subprocess.call(cmd, shell=True)

    for missing_transformer in MISSING_TRANSFORMERS:
        cmd = 'touch {}/{}'.format(transformer_dir, missing_transformer)
        subprocess.call(cmd, shell=True)

    cmd = 'cp neptune.yaml temporary_neptune.yaml'.format(checkpoints_dir, transformer_dir)
    subprocess.call(cmd, shell=True)

    cmd = 'cp neptune.yaml temporary_neptune.yaml'.format(checkpoints_dir, transformer_dir)
    subprocess.call(cmd, shell=True)

    with open("temporary_neptune.yaml", 'r+') as f:
        doc = yaml.load(f)
        doc['parameters']['experiment_dir'] = temp_inference_dir

    with open("temporary_neptune.yaml", 'w+') as f:
        yaml.dump(doc, f, default_flow_style=False)

    cmd = 'neptune run --config temporary_neptune.yaml -- evaluate -p unet_weighted_padded'
    subprocess.call(cmd, shell=True)

    cmd = 'rm temporary_neptune.yaml'
    subprocess.call(cmd, shell=True)

    cmd = 'rm -rf {}'.format(temp_inference_dir)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
