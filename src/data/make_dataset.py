# -*- coding: utf-8 -*-
import os
import click
import kaggle
import logging
from pathlib import Path

DATASET_NAME = 'city-of-seattle/sea-building-energy-benchmarking'


def download_dataset(name, output_filepath, **kwargs):
    """Download the dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Start download, this can take a while...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET_NAME,
                                      path=output_filepath,
                                      unzip=True)


def init_data_dir(filepath):
    """Create empty dir if they not exists"""
    logger = logging.getLogger(__name__)
    logger.info("Init data directories")
    root_dir = '/'.join(filepath.split('/')[:-1])
    data_dirs = ['external',
                 'interim',
                 'processed',
                 'raw']
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
        for dir in data_dirs:
            os.mkdir(os.path.join(root_dir, dir))
    else:
        logger.info('already exists')


def make_data(input_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Making the dataset.')
    init_data_dir(input_filepath)
    download_dataset(DATASET_NAME, output_filepath=input_filepath)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    make_data(input_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
