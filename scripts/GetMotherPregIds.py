"""
The script identifies the pregnancy ids and connects them with the mother id in  a dictionary.
This means that a single mother id can be connected to several pregnancy ids. The data is stored
in the json format in a dictionary on the form
{'motherID_1': [pregnancy_ID_1, ..., pregnancy_ID_n], 'mother_ID_2': [pregnancy_ID_1, ..., pregnancy_ID_m], ...}
"""

import numpy as np
import pandas as pd
import hjson


# Filepath to the file containting the information
filepath = "/tsd/p805/data/durable/data/moba/Original files/sav/PDB2601_SV_INFO_v12.sav"

# Read the data into a pandas dataframe
df_ids = pd.read_spss(filepath)

# Identifying the pregnancy- and mother id columns
preg_ids = df_ids["PREG_ID_2601"].values
mom_ids = df_ids["M_ID_2601"].values

# Empty dict to store the data
moms = {}

for idx1, mom_id1 in enumerate(mom_ids):
    print("{}/{}".format(idx1 + 1, len(mom_ids)), end="\r")

    if mom_id1 in moms:
        pass
    else:
        moms[mom_id1] = []

    for idx2, mom_id2 in enumerate(mom_ids):
        if mom_id1 == mom_id2:
            if preg_ids[idx1] == preg_ids[idx2]:
                moms[mom_id1].append(preg_ids[idx1])
            else:
                moms[mom_id1].append(preg_ids[idx1])
                moms[mom_id1].append(preg_ids[idx2])


for key, values in moms.items():
    new_values = np.unique(values)
    int_values = [new_values[i].item() for i in range(len(new_values))]
    moms[key] = list(int_values)

with open("mother_pregnancy_ids.json", "w") as f:
    hjson.dump(moms, f)
