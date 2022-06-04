import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys


def find_unique_idxs(questionnaires, filename, extension="csv", return_list=True):
    """Find the unique pregnancy ids that are present in questionnaires list and writes them to a file

    Args:
        questionnaires: list over all questionnaires files with path
        filename: filename with .npy extension and path
        extension: string of extension of filename to the questionnaires
        return_list: bool, if true returns the list of shared indices

    """
    # Empty list to hold numpy arrays of the pregnancy ids
    id_arrays = []

    # Extract the pregnancy ids in each of the questionnaires
    for i in range(len(questionnaires)):
        print(
            "Extracting ids from {} out of {} questionnaires".format(
                i + 1, len(questionnaires)
            ),
            end="\r",
        )

        if extension == "csv":
            df = pd.read_csv(questionnaires[i])
        elif extension == "spss":
            df = pd.read_spss(questionnaires[i])
        preg_ids = df["PREG_ID_2601"].to_numpy()
        id_arrays.append(preg_ids)

    unique_ids = []

    # Place the first array into an independent variable, and remove it from list
    first_array = id_arrays.pop(0)

    # Initalize iterator
    iterator = tqdm(first_array, desc="")

    # Identify the ids that are present in all of the questionnaires
    print(
        "Identifying the indices that are present in all of the {} questionnaires".format(
            len(questionnaires)
        )
    )
    for val in iterator:

        counter = 0

        # Iterate through the arrays storing the ids
        for array in id_arrays:

            if val in array:
                counter += 1

                # Identify the index of val
                index = np.where(array == val)

                # Remove the value to shorten the arrays length
                array = np.delete(array, index)
            else:
                break

            if counter == len(id_arrays):
                unique_ids.append(val)

    # Save file to binary numpy file
    np.save(filename, unique_ids)

    if return_list:
        return unique_ids


def find_mothers(values):
    """Finds mothers present from a list of pregnancy ids.

    The purpose of the function is that the pregancy id list, the <values> argument, is collected from several questionnaires,
    and the function thus finds all of the mothers that participated in all of the questionnaires.

    Args:
        values: list of pregnancy ids

    Returns:
        shared_values: a numpy array containing the mother ids
    """

    # Path to directory that holds the spss files
    dirpath = "/tsd/p805/data/durable/data/moba/Original files/sav/"

    # Filepath to the file
    mom_id_file = dirpath + "PDB2601_SV_INFO_v12.sav"

    # Dataframe
    mom_df = pd.read_spss(mom_id_file)

    # Mother and pregnancy ids from the complete set of participants
    m_ids = mom_df["M_ID_2601"].to_numpy()
    p_ids = mom_df["PREG_ID_2601"].to_numpy()

    shared_values = []

    # Iterating through the values list
    for idx, i in enumerate(values):

        # Updating the progress to the terminal window
        print("{}/{}".format(idx + 1, len(values)), end="\r")

        for num, j in enumerate(p_ids):
            if i == j:
                shared_values.append(m_ids[num])
                break

    return np.array(shared_values)


def find_nans(df, variables, variable_keys):
    """The function finds the row indices of NaN values in the specified columns in a pandas DataFrame

    Args:
        df: Pandas DataFrame
        variables: Nested dict containing specified variables for selected questionnaires
        variable_keys: list holding the keys, of which the values corresponds with the column names,
                       you want to check

    Returns:
        indices: numpy array of row indices that contain NaN values

    """

    # List to hold the indices
    indices = []

    # Iterating through the variables dict
    for Qs, nested_dict in variables.items():

        # Iterating through the variables for each questionnaire
        for key, values in nested_dict.items():

            # Extracting the columns of interest
            if key in variable_keys:

                # Identify the row indices containing NaN values
                nan_rows = df[df[values].isna()].index

                # Append to incides list
                indices.append(nan_rows)

    # Turning list of indices into a numpy array
    indices = np.concatenate(indices, axis=0)

    return indices


def set_values_nan(df, val, columns, lower=True):
    """
    Function that sets specific row-values to NaN if 
    lower/higher than a value in specified columns in a pandas dataframe inplace

    Args:
        df: pandas dataframe
        val: specific numerical value to check
        columns: list with column names in the dataframe 
        lower: if True the function checks whether the row-value is lower than val
    
    Returns:
        count: integer of how many values were set to NaN
    """

    count = 0

    if isinstance(columns, pd.core.indexes.base.Index) or isinstance(columns, list):

        for col in columns:
            if lower:
                rows = np.where((df.loc[:,col].values < val) == True)[0]
            else:
                rows = np.where((df.loc[:,col].values > val) == True)[0]

            count += len(rows)
            for row in rows:
                df.loc[row,col] = np.nan 

    elif isinstance(columns, str):

        if lower:
            rows = np.where((df.loc[:,columns].values < val) == True)[0]
        else:
            rows = np.where((df.loc[:,columns].values > val) == True)[0]

        count += len(rows)
        for row in rows:
            df.loc[row,columns] = np.nan 
    else:
        print(f"Type {type(columns)} not recognized.")


    return count