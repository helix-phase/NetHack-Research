# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sqlite3
import pandas as pd

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


"""
Extracts full dataframe from the SQLite database and saves to a CSV file.

Parameters:
- dbfilename (str): Path to the SQLite database file.
- output_filename (str, optional): Path to save the output CSV file. If None, the file is not saved.

Returns:
- pandas.DataFrame: Data extracted from the database.
- variables excluded:
"""


def create_dataframe(dbfilename, output_file=None):
    # Select every variable from the database for further processing
    query = """SELECT *
    FROM games 
    ORDER BY name, starttime"""

    with sqlite3.connect(dbfilename) as conn:
        data = pd.read_sql_query(query, conn)

    # Save dataframe to the raw folder if an output filename is provided:
    if output_file:
        data.to_csv(output_file, index=False)  # saves without index column
        print("Dataframe saved to raw folder.")

    return data


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    """
    Build the database from the terminal then run this file.
    Example Usage: $ python3 build_database.py /path/to/nld-nao /path/to/output/nld-nao.db
    
    Generate Raw Dataframe:
    """
    # Full path required:
    db_path = "/code/NetHack-Research/data/raw/nld-nao.db"
    output_path = "/code/NetHack-Research/data/raw/full_data.csv"

    # Call function and save:
    create_dataframe(db_path, output_file=output_path)
