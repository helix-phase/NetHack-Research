import pandas as pd
import json


class PreProcessing:
    def __init__(self, csv_path, mapping_path):
        # Initialize with Dataframe
        self.full_data = pd.read_csv(csv_path)

        with open(mapping_path, 'r') as file:
            self.mapping = json.load(file)

    """
    Function: Convert Unix time to human-readable format and verify value types
    Parameter: None
    Return: self
    """
    def unit_conversion(self):
        # Convert time columns to human-readable datetime
        self.full_data['starttime'] = pd.to_datetime(self.full_data['starttime'], unit='s')
        self.full_data['endtime'] = pd.to_datetime(self.full_data['endtime'], unit='s')

        self.full_data['birthdate'] = pd.to_datetime(self.full_data['birthdate'], format='%Y%m%d', errors='coerce')
        self.full_data['deathdate'] = pd.to_datetime(self.full_data['deathdate'], format='%Y%m%d', errors='coerce')

        # Ensure numerical columns are stored as numeric types
        numeric_columns = ['points', 'deathdnum', 'deathlev', 'maxlvl', 'hp', 'maxhp', 'deaths', 'turns', 'realtime']
        self.full_data[numeric_columns] = self.full_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Ensure categorical columns are stored as categorical types
        categorical_columns = ['role', 'race', 'gender', 'align']
        self.full_data[categorical_columns] = self.full_data[categorical_columns].astype('category')

        print("Unit conversion completed")

        return self

    """
    Function: interprets the bitfield encoding and replaces it with natural language meaning
    Parameter: self, column for expanding, mapping dictionary
    Return: Updated self.full_data
    """
    def expand_bitfield(self):
        # Access the specific mapping dictionary (conduct_flags, achieve, flags)
        # Define columns and their corresponding mappings
        bitfield_columns = {
            "conduct": "conduct",
            "achieve": "achieve",
            "flags": "flags"
        }

        for column, mapping_key in bitfield_columns.items():
            mapping = self.mapping[mapping_key]

            # Replace the bitfield encoding with language descriptions:
            for idx, value in self.full_data[column].items():
                try:
                    if isinstance(value, str) and value.startswith("0x"):
                        # Convert the hex string to an integer
                        value = int(value, 16)
                    else:
                        # Or convert to integer directly
                        value = int(value)

                    # Decode the bitfield by finding matching bits in the mapping dictionary
                    decoded = []
                    for bit, name in mapping.items():
                        if value & int(bit, 16):
                            decoded.append(name)

                    # Join the decoded names into a single string
                    decoded = ",".join(decoded)
                except (TypeError, ValueError):
                    decoded = None

                self.full_data.at[idx, column] = decoded

            print(f"Bitfield expansion for complete for column: {column}")

        return self

    """
    Function: Column re-ordering for ease of analysis
    Parameter: None
    Return: reordered dataframe
    """
    def reorder_columns(self):
        first_columns = ['name', 'starttime', 'endtime']
        other_columns = [col for col in self.full_data.columns if col not in first_columns]

        reordered_columns = first_columns + other_columns

        self.full_data = self.full_data[reordered_columns]

        print("Column Reordering Completed")

        return self

    """
    Function: Process dataframe and to save file
    Parameter: output path
    Return: None
    """
    def save_to_file(self, output_path):
        # Process Data
        self.unit_conversion()
        self.expand_bitfield()
        self.reorder_columns()

        self.full_data.to_csv(output_path, index=False)

        print(f"Data Saved to {output_path}")


if __name__ == "__main__":
    preprocess = PreProcessing(csv_path="/code/NetHack-Research/data/raw/full_data.csv",
                               mapping_path="/code/NetHack-Research/references/bitfield_mapping.json")

    output = "/code/NetHack-Research/data/processed/processed_data.csv"
    preprocess.save_to_file(output)
