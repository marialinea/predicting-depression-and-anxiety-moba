"""
The script separates a dataframe containing data from several questionnaires into
dataframes for each specific questionnaire after the preparing-procedure is done.
"""

import numpy as np 
import pandas as pd
import os
import pdb

from utils.dataframe import variable_permutation



###################### Parameters to specify ###########################


# List of the questionnaires
Q_names = ["Q1", "Q3", "Q4", "Q5", "Q6"]


# Target preprocessing procedure
target_preprocess = "mean"

analysis = "aggregated"

# Path to dataframe folder
dataframe_path = "./../experiments/" + "_".join(Q_names) + "/dataframes/" + target_preprocess +"_scl/"

# Path to where to save the dataframes for each qustionnaire
# outpath = dataframe_path + "single_questionnaires/"
outpath = "./../experiments/" + "_".join(Q_names) + f"/{analysis}_analysis/dataframes"

if not os.path.exists(outpath):
    os.mkdir(outpath)


# prefix = "long_aggregated_{partition}_imputed_cleaned"
# prefix = "aggregated_{partition}_imputed_cleaned"
# prefix = "long_aggregated_{partition}"
prefix = "long_{partition}_aggregated_imputed_cleaned"

#########################################################################


# Keys to the variables that are in the sub dataframe
variable_keys = [
    "scl",
    "LTHofMD",
    "RSS",
    "ss",
    "edu",
    "imm",
    "abuse",
    "income",
    "GSE",
    "anger",
    "RSES",
    "SWLS",
    "ALE",
    "birth"
]


def get_name_no_qs(prefix, partition, variable_keys):
    return prefix.format(partition=partition) + "_sub_dataframe_" + "_".join(variable_keys)


def split_dataframe(train_test_dict, Q_names, dataframe_path):
    """
    Function that split a dataframe containing data from several questionnaires
    into dataframes containing only data from one questionnaire if the dataframe
    is in long format. Requires that the variables are aggregated.

    Args:
        train_test_dict: dictionary with keys 'train' and 'test'. Their respective values are the name of the dataframes
        Q_names: list over questionnaires present in the original dataframs
        dataframepath: str, path to where the dataframes are stored
    """

    for key in train_test_dict.keys():
        print(f"Partition: {key}")

        partition_name = train_test_dict[key]

        df = variable_permutation(partition_name, dataframe_path, return_df=True)
        
        for i, Q in enumerate(Q_names):
            print(f"{i+1}/{len(Q_names)}", end="\r")
            new_name = get_name_no_qs(prefix, key, variable_keys) + f"_{Q}.csv"
        
            new_df = df.iloc[i::len(Q_names),:]

            if Q != "Q4" and "birth" in variable_keys:
                birth_items = ["birth_w", "birth_comp", "birth_exp"]

                columns = list(df.columns)

                [columns.remove(b) for b in birth_items]

                new_df = new_df[columns]
                
        
            new_df.to_csv(os.path.join(outpath, new_name), index=False)

train_test_dict = {
    "train" : get_name_no_qs(prefix, "train", variable_keys) + "_" + "_".join(Q_names) + ".csv",
    "test"  : get_name_no_qs(prefix, "test", variable_keys) + "_" + "_".join(Q_names) + ".csv"
}


split_dataframe(train_test_dict, Q_names, dataframe_path)


# for key in train_test_dict.keys():
#      print(f"Partition: {key}")
     
#      for i, Q in enumerate(Q_names):
        
#         new_name = get_name_no_qs(prefix, key, variable_keys) + f"_{Q}.csv"
        

#         df = pd.read_csv(outpath + new_name)

#         if Q == "Q4":
#             pdb.set_trace()







