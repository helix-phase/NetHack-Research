# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# NetHack Environment:
import nle.dataset as nld
from nle.dataset import db

"""
Builds the NLD-NAO database from unzipped game data files and saves it to location.

Parameters:
- nld_nao_path (str): Path to the directory containing the unzipped NLD-NAO data files.
- output_filepath (str): Path to save the generated SQLite database file.

Returns:
- None. The function performs operations in place and logs progress.

Example Usage:
$ python3 build_database.py /path/to/nld-nao /path/to/output/nld-nao.db
"""


@click.command()
@click.argument('nld_nao_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def build_database(nld_nao_path, output_filepath):
    # Logs progress and information using Python's logging module
    logger = logging.getLogger(__name__)
    logger.info('Starting Database Creation Process')

    # Check if the database already exists:
    if not nld.db.exists(output_filepath):
        logger.info("NLD-NAO does not exist. Creating...")

        nld.db.create(output_filepath)

        # Add NLD-NAO data, use the `add_altorg_directory` function:
        nld.add_altorg_directory(nld_nao_path, "nld-nao", output_filepath)
        logger.info("NLD-NAO Database Successfully Created.")
    else:
        logger.info("NLD-NAO already exists. Skipping Creation.")

    # Verify NLD-NAO Database Content with Game Count:
    db_conn = nld.db.connect(filename=output_filepath)
    game_count = nld.db.count_games('nld-nao', conn=db_conn)
    logger.info(f"NLD-NAO dataset contains {game_count} games.")

    logger.info("Creation Process Complete.")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Build the Database:
    build_database()
