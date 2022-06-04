import pandas as pd
import os
import sys
from tqdm import tqdm
from time import time
import hjson
import numpy as np
import pdb
import re

from .data import get_selected_columns
from .find import find_mothers


def load_dataframe(framename, data_path, prefix_df=None):
    """Function for loading dataframes

    Args:
        framename: name of dataframe with csv extension. Expects the filename to be in the
                  format 'dataframe_{variables seperated with underscores}_{questionnaires
                  seperated with underscores}.extenson', the filename can also include words
                  before 'dataframe'. E.g 'sub_dataframe_scl_education_Q1_Q3.csv'. All variable
                  names have to be ONE word.
        data_path: relative path to were the dataframe is stored

     Returns:
        df: pandas dataframe
    """

    # Relative path to the dataframe
    framepath = os.path.join(data_path, framename)

    try:
        # Extension of the file the dataframe is stored in
        extension = framename.split(".")[-1]

        # Loading dataframe
        if extension == "csv":
            df = pd.read_csv(framepath)
        elif extension == "parquet":
            df = pd.read_parquet(framepath)
        elif extension == "pkl":
            df = pd.read_pickle(framepath)
        else:
            print(f"Expected fileformats: csv, pickle or parquet, not {extension}")
        return df

    except FileNotFoundError:

    
        # Check that the file doesn't exists in the data_path directory with another order of the variables

        # Identify which variables the sub dataframe should contain based on filename
        framename_words = framename.split("_")

        # Index for where the dataframe word is placed
        dataframe_idx = framename_words.index("dataframe")

        # Index for the first variable
        first_idx = dataframe_idx + 1

        # Index for the last variable
        last_idx = [i for i in range(len(framename_words)) if framename_words[i].startswith("Q")][0] - 1

        # List over variable keys
        variable_keys = [
            framename_words[i] for i in range(first_idx, last_idx + 1)
        ]
                
        files = os.listdir(data_path)
        
        for fn in files:
            words = re.split("\.|_", fn)
            potential_variables = words[first_idx:last_idx+1]
            prefix = words[:first_idx]
            try:
                if check_var(potential_variables, variable_keys) and prefix == framename_words[:first_idx]:
                    df = pd.read_csv(data_path + fn)
                    return df
            except IndexError:
                continue
    
        print("Dataframe with variables {} not found in <{}> \n".format(variable_keys, data_path))


        # Give option to create sub dataframe if tried to access a non-existing sub dataframe
        if "sub" in framename:
        #if framename.split("_")[0] == "sub":

            new_frame = input("Create new subframe (y/n)?  \n")

            if new_frame == "y":
                print("")
                                
                # List over questionnaires
                Q_names = framename_words[last_idx + 1 : -1]

                Q_names.append(framename_words[-1].split(".")[0])
                
                # Name of whole dataframe
                dataframe_file = (
                    ("_").join((framename_words[dataframe_idx], *Q_names))
                    + "."
                    + framename_words[-1].split(".")[-1]
                )
                
                if prefix_df is not None:
                    dataframe_file = prefix_df + dataframe_file

                # Load whole dataframe
                complete_df = load_dataframe(dataframe_file, data_path)
                
                # pdb.set_trace()
                new_df = get_sub_dataframe(
                    complete_df, variable_keys, Q_names, framename, data_path
                )
                return new_df


            elif new_frame == "n":
                sys.exit("Exiting...")

            else:

                print(f"{new_frame} not valid option, expected y or n")
                sys.exit()
        else:
            sys.exit()

def check_var(present_variables, all_variables):
    """Recursive function that identifies if all of the variables in present_variables are in all_variables, disregarding the order of the variables

    Args:
        present_variables: list of the variables you want to verify is in the list all_variables
        all_variables: list of all variables

    Returns:
        bool: True if all of the variables in present_variables are in all_variables
    """
    if len(all_variables) == 1:
        return all_variables[0] in present_variables
    elif all_variables[0] in present_variables:
        return check_var(present_variables, all_variables[1:])
    else:
        return False

def get_sub_dataframe(df, variable_keys, Q_names, outfile, data_path):

    # Read dict that holds variables of interest
    with open("./data/variables.hjson", "r") as f:
        variables = hjson.load(f)

    # ---------------- Creating new dataframe ----------------- #

    print("Creating new sub dataframe")
    print("----------------------------\n")

    t0 = time()

    new_df, keys_present = get_selected_columns(df, variables, variable_keys, Q_names)

    # List over variables not present in the variables dict
    unused_varaibles = []

    # Check if any of the variables are not present in the new dataframe
    for var in variable_keys:
        if var not in keys_present:

            # If not in new dataframe append to list
            unused_varaibles.append(var)

    # Change outfile name not to include unused variables
    for var in unused_varaibles:
        outfile = outfile.replace(var + "_", "")

    print("Done in {:.1f}s \n".format(time() - t0))

    # ---------- Excluding women with more than one pregnancy ------ #

    # print("Identifying and excluding women with more than one pregnancy")
    # print("-------------------------------------------------------------\n")

    # t0 = time()

    # # Filename of file containing all pregnancy ids in the questionnaires in Q_names
    # preg_id_filename = "preg_ids_" + ("_").join(Q_names) + ".npy"

    # # Filename of mother ids file
    # mother_id_file = "moms_id_" + ("_").join(Q_names) + ".npy"

    # # Path to the id data
    # id_path = "./data/ids/"

    # # Check if mother ids file exists
    # if not os.path.exists(id_path + mother_id_file):

    #     # Loading pregnancy ids
    #     preg_ids = np.load(id_path + preg_id_filename)

    #     # Find mothers
    #     mom_ids = find_mothers(preg_ids)

    #     # Save the mother ids to file
    #     np.save(id_path + mother_id_file, mom_ids)
    # else:

    #     # Loading mother ids
    #     mom_ids = np.load(id_path + mother_id_file)

    # # Dict containing all mothers and their pregnancies in the MoBa study
    # with open(id_path + "mother_pregnancy_ids.json", "r") as f:
    #     mother_pregnancies_dict = hjson.load(f)

    # # List over pregnancies that are not the first one for every woman
    # rm_pregnacies = []

    # # Identifying pregnancy ids that belongs to the same mother
    # for mother, pregs in mother_pregnancies_dict.items():
    #     if mother in mom_ids and len(pregs) > 1:
    #         rm_pregnacies.extend(pregs[1:])

    # # Identifying the indices of the multiple pregnancies
    # rm_indices = new_df.loc[new_df["PREG_ID_2601"].isin(rm_pregnacies)].index

    # # Removing all rows belonging to the excess pregnanceis
    # new_df.drop(axis=0, index=rm_indices, inplace=True)

    # print("Removed {} pregnanices \n".format(len(rm_indices)))
   
    # print("Done in {:.1f}s \n".format(time() - t0))

    # ----------------------------------------------

    print("Saving new dataframe")
    print("----------------------\n")

    t0 = time()

    # Relative path to new dataframe
    outpath = os.path.join(data_path, outfile)

    print(f"Shape of new dataframe: {new_df.shape}")
    # Writing new dataframe to csv
    new_df.to_csv(outpath, index=False)

    print("Done in {:.1f}s \n".format(time() - t0))

    return new_df


def merge_dataframes(dfs, column_name, shared_values):
    """Merge list of pandas dataframes into one for shared values in a column

    The function merges all of the columns of several dataframes into one DataFrame
    for all identical values in a give column

    Args:
        dfs: list of pandas dataframe
        column_name: name of the column for which the dataframes shares some values
        shared_values: list of identical values in column_name for all of the dataframes in dfs

    Returns:
        new_df: new dataframe containing all of the columns for the dataframes in dfs

    Warning: The script doesn't verify that the values in the shared_values array are in fact shared among all the dfs, it expect the array to be complete
             and to not contain any values that aren't shared.

    """

    new_rows = {}

    # Iterate through the dataframes
    for i, df in enumerate(dfs):

        print(
            "Iterating through dataframe {} out of {}".format(i + 1, len(dfs)), end="\r"
        )

        # Rows containing the shared values in column <column_name>
        rows = df[df[column_name].isin(shared_values)]

        # New variable holding the remaining columns without the column <column_name>
        rows = rows.loc[:, rows.columns != column_name].to_dict(orient="records")

        for val, row in tqdm(zip(shared_values, rows), leave=False):

            # Adds the variables to the new_rows dict
            try:
                for key, values in row.items():
                    new_rows[val][key] = values
            except KeyError:
                new_rows[val] = row

    """
    # Define iterator for shared values
    val_iter = tqdm(shared_values, desc='Iterating through shared values', initial=1, total=len(shared_values), leave=False)

    # Iterate through the dataframs
    for i, df in enumerate(dfs):
        print('Iterating through df {} out of {}'.format(i+1, len(dfs)))
        
        
        # Iterate through the shared values
        for value in val_iter:

            # Whole row of the dataframe containing all of the columns
            row = df.loc[df[column_name]==value]

            # Doesn't enter the if block if value is not in df[column_name]
            if len(row[column_name]) > 0:

                # New variable holding the remaining columns
                variables = row.loc[:, row.columns != column_name].to_dict(orient='records')[0]

                # Adds the variables, either for the first time or adding them, to the new_rows dict
                try:
                    for key, values in variables.items():
                        new_rows[value][key] = values
                except KeyError:
                    new_rows[value] = variables
    """
    new_df = pd.DataFrame(new_rows).T
    new_df.index.name = column_name
    return new_df


def wide_to_long(df_wide, unique_variables, questionnaires):
    """Function that transforms a dataframe in wide format to long format

    Function expects that all the time dependent variables contain the name of the questionnaires  in capital letters in the column, e. g. mean_scl_Q1. Convert the questionnaire name into number of weeks since gestation, and place them in a new column.

    Args:
        df_wide: Pandas DataFrame in the wide format
        unique_variables: List of strings of the unique variables in the dataframe
        questionnaires: List of questionnaires present in the dataframe in captial letters
    Returns:
        df_long: Pandas DataFrame in long format

    """

    # Empty list to hold each new row in the long format dataframe
    new_rows = []

    # List over column names, except the PREG_ID_2601 column
    columns = list(df_wide.columns[1:])

    ###############################################################
    # Adding the time dependent variables to the long dataframe  #
    ###############################################################

    print("Adding the time dependent variables to the long dataframe \n")

    # Iterating over each row in the wide dataframe
    for i in tqdm(range(df_wide.shape[0]), desc="Rows"):

        # Holds the values for each column for the specific row
        row = df_wide.iloc[i]

        # First value is the PREG_ID_2601
        id = row[0]

        # Looping over each questionnaire
        for q in questionnaires:

            # Every questionnaire for every unique id has its own dict
            q_row = {}
            q_row["PREG_ID_2601"] = id
            q_row["Time"] = q

            # Iterationg over the remaining columns
            for col_name in row.index[1:]:

                # If the questionnaire name is in the column name, add its value to the dict
                if q in col_name:

                    # The unique variable name, i. e. excluding the questionnaire ending of the name
                    new_name = "_".join(col_name.split("_")[:-1])

                    # Adding the column value for the time dependent variable
                    q_row[new_name] = row[col_name]
                    try:
                        # Crossing the variable of in the column list, every variable that remains after the outer loop is a time independent variable
                        columns.remove(col_name)
                    except ValueError:
                        continue

            # Appending the dictionary to the row list
            new_rows.append(q_row)

    # Formatting the new_rows list so that we can transform it into a dataframe
    df_long = pd.DataFrame.from_dict(new_rows)

    ################################################################
    # Adding the time independent variables to the long dataframe  #
    ################################################################

    print("Adding the time independent variables to the long dataframe \n")

    # Empty list to hold each new column for time independent variables
    new_columns = []

    # Iterating over the remaining variables from the wide dataframe
    for col in tqdm(columns, desc="Number of independent variables added to dataframe"):

        # Each time independent variable has its own dict to store its values
        new_column = {}
        new_column[col] = []

        # Iterating over all of the unique values in the column
        for i in range(len(df_wide[col])):

            # Every value repeats it self as many times as the number of questionnaires in the long format
            for j in range(len(questionnaires)):

                # Append column value
                new_column[col].append(df_wide[col].iloc[i])

        # Append whole new column to new_columns list
        new_columns.append(new_column)

    # Append each new column to the long dataframe
    for i in range(len(new_columns)):
        dict = new_columns[i]
        df_dict = pd.DataFrame(dict)
        df_long = pd.concat([df_long, df_dict], axis=1)

    def transform_Q(Q):
        """Function that transform questionnaire to number of weeks since gestation

        Args:
            Q: String containing questionnaire name in the format 'QX'
        Returns:
            Number of weeks since gestation
        """
        qs_to_time = {"Q1": 15, "Q2": 22, "Q3": 30, "Q4": 64, "Q5": 112, "Q6": 184}

        return qs_to_time[Q]

    print(
        "Transforming the time variable from questionnaire to weeks since gestation \n"
    )

    # Transforming the questionnaire names to the time they were administered
    df_long["Time"] = df_long["Time"].transform(transform_Q)

    return df_long

def variable_permutation(dataframe_file, data_path, return_df=True):
    """
    Function that check that checks that the dataframe file doesn't exists in the 
    data_path directory with the variables stacked in a different order.

    Args:
        dataframe_file: filename of the dataframe with file extension
        data_path: path to directory or path to specific file.
        return_df: bool, default True. If True and the dataframe exists, the dataframe is returned. If False, a boolean value is returned.
                   if dataframe exists, True is returned. False if not exists. 

    Returns:
        See return_df arg.
    """

    # Identify which variables the sub dataframe should contain based on filename
    dataframe_file_words = dataframe_file.split("_")

    # Index for where the dataframe word is placed
    dataframe_idx = dataframe_file_words.index("dataframe")

    # Index for the first variable
    first_idx = dataframe_idx + 1

    # Index for the last variable
    last_idx = [i for i in range(len(dataframe_file_words)) if dataframe_file_words[i].startswith("Q")][0] - 1

    # List over variable keys
    variable_keys = [
        dataframe_file_words[i] for i in range(first_idx, last_idx + 1)
    ]
    
    if os.path.isdir(data_path) is True:

        files = os.listdir(data_path)
    
    elif os.path.isfile(data_path) is True:

        files = [data_path]

    else:
        
        return False


    for fn in files:
        words = re.split("\.|_", fn)
        potential_variables = words[first_idx:last_idx+1]
        prefix = words[:first_idx]
        try:
            if check_var(potential_variables, variable_keys) and prefix == dataframe_file_words[:first_idx]:
                df = pd.read_csv(data_path + fn)
                if return_df is True:
                    return df
                else: 
                    return True
        except IndexError:
            continue

    return False