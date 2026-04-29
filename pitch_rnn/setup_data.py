"""
Docstring for model_training_notebooks.src.setup_data

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

from model_shared.db import query_table_for_features
from model_shared.feature_list import validate_feature_list_file

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