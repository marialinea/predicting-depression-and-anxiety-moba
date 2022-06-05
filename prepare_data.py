import numpy as np
import pandas as pd
import os
import sys
import hjson
import json
from time import time
import pdb

from utils.dataframe import wide_to_long
from utils.prepare import PrepareData

####################### Loading variables ############################
t0 = time()

# Read dict that holds variables of interest
with open("./data/variables.hjson", "r") as f:
    variables = hjson.load(f)


###################### Parameters to specify ###########################


# List of the questionnaires
# Q_names = ["Q1", "Q3", "Q4", "Q5", "Q6"]
Q_names = ["Q1", "Q3", "Q4"]


# Path to experiment folder, i. e. different combinations of questionnaires are one experiment and thus have their own folder
data_path = "./experiments/" + "_".join(Q_names) + "/wide_item_analysis/dataframes/"

# Name of dataframe (if exists)
df_name = "test_sub_dataframe_scl_LTHofMD_RSS_ss_edu_imm_abuse_income_GSE_anger_RSES_SWLS_Q1_Q3_Q4.csv"
# Target preprocessing procedure
target_preprocess = "mean"

# Make sure all NaN values are correctly encoded
clean = True

# Impute NaN values
impute = False

# Aggregate variables
aggregate = True

# Save prepared dataframes
save = True

#######################################################################

# Path to store the preprocessed dataframes
out_path = data_path + target_preprocess + "_scl/"


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
"""
variable_keys = [
    "HeightWeight",
    "PreviousPregnany",
    "Edu",
    "Work",
    "StrainsWork",
    "Household",
    "Language",
    "IncomeHousing",
    "Alcohol",
    "EatingDisorder",
    "SWLS",
    "RSS",
    "SocialSupport",
    "scl",
    "Abuse",
    "RSES",
    "Work_Q3",
    "SickLeave_Q3",
    "Lifting",
    "Drugs",
    "CivilStatus",
    "Emotion",
    "AdverseLifeEvents",
    "Assault_Q3",
    "Assault_Q6",
    "Birth",
    "ChildDevelopment_Q4",
    "ChildDevelopment_Q5",
    "ChildDevelopment_Q6",
    "ChildCare",
    "ChildMood",
    "Finance",
    "Smoking_Q4",
    "EPDS",
    "ChildLengthWeight",
    "Communication",
    "ChildTemperament",
    "ChildBehaviour",
    "Daycare",
    "LivingWithFather",
    "LivingSituation",
    "LivingEnvironment",
    "TimeOutside",
    "PregnantNow",
    "ParentalLeave",
    "Smoking_Q5",
    "WHOQOL",
    "Autism",
    "WalkUnaided",
    "SocialSkills",
    "SocialCommunication",
    "Work_Q6",
    "AdultADHD",
    "PLOC"
]

"""
print("Start preparing data \n")

# Calling PreparingData class



prepared = PrepareData(variables, variable_keys, Q_names, target_preprocess, data_path, out_path=data_path, clean=clean, impute=impute, aggregate=aggregate, save=save, df_name=df_name, split=False)


################## Convert from wide to long format ######################

# # Name of train df long format
# train_name = prepared.out_path + "long_" + prepared.train_name.split("/")[-1]

# # Name of test df long format
# test_name = prepared.out_path + "long_" + prepared.test_name.split("/")[-1]


# if os.path.exists(train_name) and os.path.exists(test_name):
#     # Read dfs from file
#     df_long_train = pd.read_csv(train_name)
#     df_long_test = pd.read_csv(test_name)

# else:

#     df_train = prepared.train_test_dict["train"]["df"]
#     df_test = prepared.train_test_dict["test"]["df"]

#     # Converting from wide to long
#     df_long_train = wide_to_long(df_train, prepared.unique_variables, Q_names)
#     df_long_test = wide_to_long(df_test, prepared.unique_variables, Q_names)

#     # Saving dfs
#     df_long_train.to_csv(train_name, index=False)
#     df_long_test.to_csv(test_name, index=False)


# print("Done in {:.1f}s \n".format(time() - t0))
