"""

Script making one dataframe from two timepoints, time dependent variables are combined into one variable where
the new variable is the difference between the two time points

"""


import pandas as pd
import numpy as np
import pdb
import re


target_preprocessing = "mean" # or "sum"
dataframe_file = "test_preprocessed_sub_dataframe_{}_scl_LTHofMD_RSS_ss_edu_imm_income_SWLS_ALE_Q1_Q4.csv".format(target_preprocessing)

data_path = "../dataframes/{}_scl/".format(target_preprocessing)

# Loading dataframe
df = pd.read_csv(data_path + dataframe_file)

# New dataframe 
delta_df = pd.DataFrame()

# All time dependent variables has this name pattern
time_questions_pattern = r"(.*)_(Q\d{1})"

# Lists over time independent and depedent variables
time_independent = []
time_dependent = []

# List over all variables in the dataframe
variables = list(df.columns)

# Iterating over all varaibles to identify time dependence
for var in variables:
    if re.search(pattern=time_questions_pattern, string=var) != None:
        time_dependent.append(var)
    else:
        time_independent.append(var)

# Adding time independent variables
delta_df[time_independent] = df[time_independent]

# Copy of time dependent variables
tmp_time_dependent = time_dependent.copy()


for var in time_dependent:

    if var not in tmp_time_dependent:
        continue

    name = '_'.join(var.split('_')[:-1])
    timepoint1 = (var.split('_')[-1])[-1]
    
    tmp_time_dependent.remove(var)

    # If there is a matching variable at a later time point, enter if test
    if list(filter(lambda x: name in x, tmp_time_dependent)):

        # Name pattern of matching variable
        var_pattern = name + r"_Q(\d{1})"

        # Iterating over remaining variables
        for time_var in tmp_time_dependent:

            if re.search(pattern=var_pattern, string=time_var) != None:

                tmp_time_dependent.remove(time_var)

                timepoint2 = (time_var.split('_')[-1])[-1]

                new_col = "delta_" + name

                if timepoint1 > timepoint2:
                    delta_df[new_col] = df[var] - df[time_var]
                else:
                    delta_df[new_col] = df[time_var] - df[var]
                
                break


#print(delta_df.head())

delta_df.to_csv(data_path + "delta_" + dataframe_file, index=False)

