import numpy as np
import pandas as pd
import os
import sys
import hjson
import pdb

from utils.dataframe import load_dataframe


# Read dict that holds variables of interest
with open("./data/variables.hjson", "r") as f:
    variables = hjson.load(f)

# Keys to the variables that are in the sub dataframe
variable_keys = [
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
    "ALE"
]


# Q_names = ["Q1", "Q3", "Q4", "Q5", "Q6"]
Q_names = ["Q1", "Q3", "Q4"]

prefix_df = "test_imputed_cleaned_"

# Path to experiment folder, i. e. different combinations of questionnaires are one experiment and thus have their own folder
data_path = "./experiments/" + "_".join(Q_names) + "/wide_format_analysis/dataframes/"


var_qs_name = (
             "_".join(variable_keys)
            + "_"
            + "_".join([Q for Q in variables.keys() if Q in Q_names])
            + ".csv"
        )

# Name of the finale dataframe
unprocessed_name = "sub_dataframe_" + var_qs_name

# Creating dataframe
df = load_dataframe(unprocessed_name, data_path, prefix_df=prefix_df)
