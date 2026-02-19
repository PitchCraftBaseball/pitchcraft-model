"""
Docstring for model-training-notebooks.src.setup_data

setup_data.py takes the beginning of the RNN training notebook (data prep)
and turns it into a setup script. The proposed workflow is 

setup_data.py <feature_list> -> rnn_data.csv 
model_rnn_implementation sources rnn_data.csv the way that it currently does

<feature_list> is some language agnostic file that takes in table features
you can find valid table features here: https://github.com/PitchCraftBaseball/statcast-etl#

the training data (as of 2/8/2026) only pulls from the historical table for data.
we should probably talk about that 
"""

import csv
from pathlib import Path
import sys


from model_shared.db import find_table_for_column, query_table_for_features

"""
    Validates the fields that are passed by the feature_list file. 
    Validation fails if at least one of the fields is incorrect.
    This function assumes that we might query from non-historical data sometime
    in the future. 
"""
def validate_feature_list_file(filename: str):
    count: int = 0
    table_to_features_map: dict[str, list[str]] = {}
    with open(filename, "r") as file: 
        for feature in file:
            if (feature[0] == "#"): # * Support for comments in the feature file
                # print(f"{feature} is currently disabled")
                continue
            feature = feature.strip()
            table_name = find_table_for_column("public", feature)
            if table_name is None:
                print(f"Couldn't find a table that contained the feature: {feature}")
                return None
            count += 1
            if table_name in table_to_features_map:
                table_to_features_map[table_name].append(feature)
            else: 
                table_to_features_map[table_name] = [feature]
    print(f"<feature_file> successfully validated. Feature count: {count}")
    return table_to_features_map

# we'll want to flush everything to CSV files. For now, I'll write it to multiple
# by table because I'm not entirely sure if we use columns from other tables
if __name__ == "__main__":
    # checking feature_list file to see if each column actually exists 
    feature_list_file = sys.argv[1]
    feature_name_list = validate_feature_list_file(feature_list_file)
    if feature_name_list: 
        for table_name, feature_list in feature_name_list.items(): 
            print("Querying table")
            cursor_result = query_table_for_features(table_name, feature_list)
            print("Writing to file...")
            with open(f'{table_name}_rnn_data.csv', 'w', newline='') as csvFile: 
                writer = csv.writer(csvFile) 
                writer.writerow(feature_list)
                writer.writerows(cursor_result)
                print(f'Features written to {table_name}_rnn_data.csv')
                
    print("Exiting")