"""
The scripts separates the items in a dataframe containing item level data from 
several questionnaires into constant items and time varying items. The constant items
are then placed into the dataframes for all of the questionnaires, while the time 
dependent items are only placed in the dataframe corresponding to their time point (questionnaire).
"""

import hjson
import pandas as pd
import pdb
import os

from utils.dataframe import variable_permutation

# Read dict that holds variables of interest
with open("./../data/variables.hjson", "r") as f:
    variables = hjson.load(f)



###################### Parameters to specify ###########################


# List of the questionnaires
Q_names = ["Q1", "Q3", "Q4", "Q5", "Q6"]


# Target preprocessing procedure
target_preprocess = "mean"

analysis = "item_level"

# Path to dataframe folder
dataframe_path = "./../experiments/" + "_".join(Q_names) + "/dataframes/" + target_preprocess +"_scl/"

# Path to where to save the dataframes for each qustionnaire
# outpath = dataframe_path + "single_questionnaires/"
outpath = "./../experiments/" + "_".join(Q_names) + f"/{analysis}_analysis/dataframes/"

if not os.path.exists(outpath):
    os.mkdir(outpath)


# prefix = "long_aggregated_{partition}_imputed_cleaned"
# prefix = "aggregated_{partition}_imputed_cleaned"
# prefix = "long_aggregated_{partition}"
# prefix = "long_{partition}_aggregated_imputed_cleaned"
prefix = "{partition}_imputed_cleaned"

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

train_test_dict = {
    "train" : get_name_no_qs(prefix, "train", variable_keys) + "_" + "_".join(Q_names) + ".csv",
    "test"  : get_name_no_qs(prefix, "test", variable_keys) + "_" + "_".join(Q_names) + ".csv"
}

def get_items(Q_names, variable_keys, variables):
    item_names = []

    # Get name of the different items without the _QX extension
    for Q in Q_names:
        Q_id = f"_{Q}"
        keys = list(variables[Q].keys())

        for key in keys:
            if Q_id not in key and key in variable_keys:
                item_names.append(key)

    constant_items = []
    time_items = []
    tmp = variable_keys.copy()

    # Find items that varies from questionnaire to questionnaire
    for i in range(len(item_names)):
        name = item_names.pop(0)
        if name not in item_names and name in tmp:
            constant_items.append(name)
            tmp.remove(name)
        elif name in item_names and name in tmp:
            time_items.append(name)
            tmp.remove(name)

    time_items[0] = "mean_scl"

    const_items = {}

    # Get single items belonging to the constant items
    for var in constant_items:
        const_items[var] = {}
        for Q in Q_names:
            keys = variables[Q].keys()
            if var in keys:
                const_items[var] = variables[Q][var]


    all_variables = {}

    # Add all of the constant items to each dataframe column-list
    for Q in Q_names:
        all_variables[Q] = []
        for key, values in const_items.items():
            if key == "birth" and Q != "Q4":
                    continue
            all_variables[Q].extend(values)



    for item in time_items:
        for Q in Q_names:
            try:
                if item == "mean_scl":
                    Q_id = f"_{Q}"
                    all_variables[Q].append(variables[Q][item + Q_id])
                else:
                    all_variables[Q].extend(variables[Q][item])
            except KeyError:
                continue

    return all_variables

all_items = get_items(Q_names, variable_keys, variables)


for key in train_test_dict.keys():
        print(f"Partition: {key}")

        partition_name = train_test_dict[key]

        df = variable_permutation(partition_name, dataframe_path, return_df=True)

        for i, Q in enumerate(Q_names):
            print(f"{i+1}/{len(Q_names)}", end="\r")
            new_name = "long_" + get_name_no_qs(prefix, key, variable_keys) + f"_{Q}.csv"

            new_df = df[all_items[Q]]

            new_df = new_df.rename(columns={f"mean_scl_{Q}": "mean_scl"})

            new_df.to_csv(os.path.join(outpath, new_name), index=False)


