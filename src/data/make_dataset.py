# -*- coding: utf-8 -*-
import os
from string import Template
import click
import logging
from pathlib import Path
import requests
from tqdm import tqdm

BASE_URL = 'https://data.seattle.gov/api/views/$id/rows.csv?accessType=DOWNLOAD'
SET_IDs = {
    2015 : 'h7rm-fz6m',
    2016 : '2bpz-gwpy'
    }

def download_dataset(url, output_filepath, **kwargs):
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

def init_data_dir(filepath):
    """Create empty dir if they not exists"""
    logging.info("Init data directories")
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
        print('already exists')

def make_data(input_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Making the dataset.')
    for name, id in SET_IDs.items():
        download_dataset(Template(BASE_URL).substitute(id=id),
                         os.path.join(input_filepath),
                         encoding='gzip')

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
    #load_dotenv(find_dotenv())

    main()
