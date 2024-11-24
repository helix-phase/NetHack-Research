import sqlite3
import pandas as pd

"""
Extracts full game dataframe from the SQLite database and saves to a CSV file.

Parameters:
- dbfilename (str): Path to the SQLite database file.
- output_filename (str, optional): Path to save the output CSV file. If None, the file is not saved.

Returns:
- pandas.DataFrame: Data extracted from the database.
- variables excluded:
"""


def game_data(dbfilename, output_file=None):
    # Selects most variables from the database
    query = """SELECT name, gameid, points, turns, maxlvl, starttime, endtime, hp, maxhp, deathlev, deathdnum, death, 
    role, race, gender, align, realtime, deaths
    FROM games 
    ORDER BY name, starttime"""

    with sqlite3.connect(dbfilename) as conn:
        data = pd.read_sql_query(query, conn)

    # Convert to Datetime:
    data['starttime'] = pd.to_datetime(data['starttime'])
    data['endtime'] = pd.to_datetime(data['endtime'])

    # Save dataframe to the raw folder if an output filename is provided:
    if output_file:
        data.to_csv(output_file, index=False)  # saves without index column
        print("Dataframe saved to raw folder.")

    return data


# Full path is required:
db_path = "/code/NetHack-Research/data/raw/nld-nao.db"
output_path = "/code/NetHack-Research/data/raw/full_data.csv"

# Call function and save:
game_data(db_path, output_file=output_path)
