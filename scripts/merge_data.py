"""

The script finds datafiles, reads them into a list pandas dataframes, identifies
common values by column, merges the dataframes into one dataframe and saves it to a csv file.

"""

import pandas as pd
import numpy as np
import os
import sys
from time import time
import pdb

from utils.data import get_files
from utils.dataframe import merge_dataframes
from utils.find import find_unique_idxs

# ---------------- Finding the datafiles ----------------- #

print("Fetching datafiles")
print("------------------------------\n")

t0 = time()


# List of strings that are present in the filenames of the questionnaires
Qs = ["_Q1_", "_Q3_", "_Q4_", "_Q5_", "_Q6_"]
# Qs = ["_Q1_", "_Q4_"]

# File extension of the files, can be csv or sav
extension = "csv"

# Remove the underscore from the questionnaire names
Q_names = [Q.replace("_", "") for Q in Qs]

# Experiment folder, i. e. different combinations of questionnaires are one experiment and thus have their own folder
exp_folder = "../experiments/" + "_".join(Q_names)

# Extension of outputfile
out_extension = ".csv"

# Name of the final outputfile
outfile = "dataframe_" + "_".join(Q_names) + out_extension

# Path to the folder that will store the dataframe
dataframe_folder = os.path.join(exp_folder, "dataframes")

# Check if the folder exists
if not os.path.exists(dataframe_folder):
    os.mkdir(dataframe_folder)

# Relative path and filename of the finale outputfile
framepath = os.path.join(dataframe_folder, outfile)


# Check if file already exists
if os.path.exists(framepath):
    sys.exit("Dataframe file found in {}".format(framepath))

# Storing the files, with absolute path, in a list
Q_files = get_files(extension, Qs)


print("Done in {:.1f}s \n".format(time() - t0))


# ---------------- Reading files into dataframes ----------------- #


print("Reading files into dataframes")
print("------------------------------\n")


t0 = time()


# List of dataframes
if extension == "csv":
    dfs = [pd.read_csv(fn) for fn in Q_files]
elif extension == "sav":
    dfs = [pd.read_spss(fn) for fn in Q_files]


print("Done in {:.1f}s \n".format(time() - t0))


# ---------------- Identifying pregnancy ids ----------------- #


print("Identifying pregnancy ids present in all questionnaires")
print("---------------------------------------------------------\n")


t0 = time()


# Name of the file that holds the shared ids
preg_id_filename = "preg_ids_" + "_".join(Q_names) + ".npy"

# Relative path to id folder
id_path = "./data/ids"


# Relative path and filename to the pregnancy id file
preg_id_path = os.path.join(id_path, preg_id_filename)


# Check if file already exists
if os.path.exists(preg_id_path):
    pregnancy_ids = np.load(preg_id_path)
else:
    pregnancy_ids = find_unique_idxs(Q_files, preg_id_path)


print("Done in {:.1f}s \n".format(time() - t0))


# ---------------- Merging dataframes ----------------- #


print("Start mergings dataframes with shared ids")
print("------------------------------------------\n")


t0 = time()


# Name of the pregnancy id column in all of the dataframes
column_name = "PREG_ID_2601"


# Merging
merged_df = merge_dataframes(dfs, column_name, pregnancy_ids)


print("Done in {:.1f}s \n".format(time() - t0))


print("Identifying and excluding women with more than one pregnancy")
print("-------------------------------------------------------------\n")

t0 = time()

Q_names = ["Q1", "Q3", "Q4", "Q5", "Q6"]

# Filename of file containing all pregnancy ids in the questionnaires in Q_names
preg_id_filename = "preg_ids_" + ("_").join(Q_names) + ".npy"

# Filename of mother ids file
mother_id_file = "moms_id_" + ("_").join(Q_names) + ".npy"

# Path to the id data
id_path = "./data/ids/"

# Check if mother ids file exists
if not os.path.exists(id_path + mother_id_file):

    # Loading pregnancy ids
    preg_ids = np.load(id_path + preg_id_filename)

    # Find mothers
    mom_ids = find_mothers(preg_ids)

    # Save the mother ids to file
    np.save(id_path + mother_id_file, mom_ids)
else:
    # Loading mother ids
    mom_ids = np.load(id_path + mother_id_file)

# Dict containing all mothers and their pregnancies in the MoBa study
with open(id_path + "mother_pregnancy_ids.json", "r") as f:
    mother_pregnancies_dict = hjson.load(f)

# List over pregnancies that are not the first one for every woman
rm_pregnacies = []

# Identifying pregnancy ids that belongs to the same mother
for mother, pregs in mother_pregnancies_dict.items():
    if mother in mom_ids and len(pregs) > 1:
        rm_pregnacies.extend(pregs[1:])


# Identifying the indices of the multiple pregnancies
rm_indices = merged_df.loc[merged_df["PREG_ID_2601"].isin(rm_pregnacies)].index


# Removing all rows belonging to the excess pregnanceis
merged_df.drop(axis=0, index=rm_indices, inplace=True)


print("Saving dataframe")
print("-----------------\n")


merged_df.to_parquet(framepath)
if out_extension == ".csv":
    merged_df.to_csv(framepath)
elif out_extension == ".parquet":
    merged_df.to_parquet(framepath)
elif out_extension == ".pkl":
    merged_df.to_pickle(framepath)
