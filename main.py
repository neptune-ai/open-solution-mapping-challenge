import click
from src.pipeline_manager import PipelineManager

pipeline_manager = PipelineManager()


@click.group()
def main():
    pass


@main.command()
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def prepare_masks(dev_mode):
    pipeline_manager.prepare_masks(dev_mode)


@main.command()
@click.option('-tr', '--train_data', help='calculate for train data', is_flag=True, required=False)
@click.option('-val', '--valid_data', help='calculate for validation data', is_flag=True, required=False)
def prepare_metadata(train_data, valid_data):
    pipeline_manager.prepare_metadata(train_data, valid_data)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train(pipeline_name, dev_mode):
    pipeline_manager.start_experiment()
    pipeline_manager.train(pipeline_name, dev_mode)
    pipeline_manager.finish_experiment()


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run evaluation on', type=int, default=None,
              required=False)
def evaluate(pipeline_name, dev_mode, chunk_size):
    pipeline_manager.start_experiment()
    pipeline_manager.evaluate(pipeline_name, dev_mode, chunk_size)
    pipeline_manager.finish_experiment()


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dir_path', help='directory with images to score', required=True)
@click.option('-r', '--prediction_path', help='path to the prediction .json file', required=True)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def predict_on_dir(pipeline_name, dir_path, prediction_path, chunk_size):
    pipeline_manager.predict_on_dir(pipeline_name, dir_path, prediction_path, chunk_size)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run evaluation and prediction on', type=int,
              default=None, required=False)
def train_evaluate(pipeline_name, dev_mode, chunk_size):
    pipeline_manager.start_experiment()
    pipeline_manager.train(pipeline_name, dev_mode)
    pipeline_manager.evaluate(pipeline_name, dev_mode, chunk_size)
    pipeline_manager.finish_experiment()


if __name__ == "__main__":
    main()
