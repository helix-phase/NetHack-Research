import pandas as pd
import json


class PreProcessing:
    def __init__(self, csv_path, mapping_path):
        # Initialize with Dataframe
        self.full_data = pd.read_csv(csv_path)

        with open(mapping_path, 'r') as file:
            self.mapping = json.load(file)

    # TODO: Write preprocessing functions:
    """
    Function: Convert Unix time to human-readable format and verify value types
    Parameter: None
    Return: self
    """

    def unit_conversion(self):
        # Convert Unix time columns to human-readable datetime
        self.full_data['starttime'] = pd.to_datetime(self.full_data['starttime'], unit='s')
        self.full_data['endtime'] = pd.to_datetime(self.full_data['endtime'], unit='s')

        # Ensure numerical columns are stored as numeric types
        columns = []

        # Ensure categorical columns are stored as dtype

        print(self.full_data)
        return self

    """
    Function: interprets the bitfield encoding and replaces it with natural language meaning
    Parameter: self, column for expanding, mapping dictionary
    Return: Updated self.full_data
    """
    def expand_bitfield(self, column):
        # Replaces the mapping with the encoded values in the same column:
        for bit, name in self.mapping.items():
            self.full_data[column] = self.full_data[column].apply(lambda x: self.mapping[x])

        return self

    """
    Function: Column re-ordering for ease of analysis
    Parameter: 
    Return: 
    """

    """
    Function: Save processed dataframe to file
    Parameter: 
    Return: 
    """


if __name__ == "__main__":
    preprocessor = PreProcessing(csv_path="/code/NetHack-Research/data/raw/full_data.csv",
                                 mapping_path="/code/NetHack-Research/references/bitfield_mapping.json")

    preprocessor.unit_conversion()
