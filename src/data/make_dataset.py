# -*- coding: utf-8 -*-
import getpass
import logging
import os
from pathlib import Path
from string import Template

import click
import requests
from tqdm import tqdm

try:
    import kaggle
except OSError:
    print("Kaggle API's credential required")
    username = input("kaggle username : ")
    key = getpass.getpass("kaggle API key")
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key


# Get data from Kaggle
DATASET_NAME = 'city-of-seattle/sea-building-energy-benchmarking'

# Get data from seattle's website
BASE_URL = 'https://data.seattle.gov/api/views/$id/rows.csv?accessType=DOWNLOAD'

SET_IDs = {
    # 2015 : 'h7rm-fz6m', # from Kaggle
    # 2016 : '2bpz-gwpy',
    2017: 'qxjw-iwsh',
    2018: '7rac-kyay'
    }


def download_dataset(name, output_filepath, **kwargs):
    """Download the dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Start download, this can take a while...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET_NAME,
                                      path=output_filepath,
                                      unzip=True)

def download_dataset_seattle(url, output_filepath, **kwargs):
    """Download the dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Start download, this can take a while...")
    user_agent = kwargs.get('user_agent', 'Mozilla/5.0')
    encoding = kwargs.get('encoding', None)
    headers = {'User-agent' : user_agent,
               'Accept-Encoding': encoding
               }
    logger.debug(headers)
    response = requests.get(url, stream=True, headers=headers)
    logger.debug("response's headers : %s", response.headers)
    content_disp = response.headers.get('Content-disposition')
    filename = content_disp.split('=')[1]
    logger.debug("filename %s", filename)
    with open(os.path.join(output_filepath, filename), 'wb') as file:
        chunk_size = 1024
        for data in tqdm(response.iter_content(chunk_size=chunk_size),
                  desc='Download data'):
                  file.write(data)

def init_data_dir(project_dir):
    """Create empty dir if they not exists"""
    logger = logging.getLogger(__name__)
    logger.info("Init data directories")
    filepath = Path(project_dir).resolve()
    data_dirs = ['external',
                 'interim',
                 'processed',
                 'raw']
    data_path = filepath.joinpath('data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        for directory in data_dirs:
            os.mkdir(data_path.joinpath(directory))
    else:
        logger.info('already exists')


def make_data(input_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Making the dataset.')
    download_dataset(DATASET_NAME, output_filepath=input_filepath)
    for name, id in SET_IDs.items():
        download_dataset_seattle(Template(BASE_URL).substitute(id=id),
                                 os.path.join(input_filepath),
                                 encoding='gzip')


@click.command()
@click.argument('input_filepath', type=click.Path())
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
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    init_data_dir(project_dir)
    main()
