## GetMotherPregIds.py
----------------------------------------------------

The script identifies the pregnancy ids and connects them with the mother id in  a dictionary.
This means that a single mother id can be connected to several pregnancy ids. The data is stored
in the json format in a dictionary on the form
{'motherID_1': [pregnancy_ID_1, ..., pregnancy_ID_n], 'mother_ID_2': [pregnancy_ID_1, ..., pregnancy_ID_m], ...}



## merge_data.py
-------------------------------------------------------
Generates a dataframe containing all the variables from several questionnaires run the `merge_data.py` script. The script identifies the pregnancy ids that are present in all of the specified questionnaires, and stores their corresponding variables in one dataframe. The pregnancy ids are stored in the `./data/ids/` folder.

Specifiy the questionnaires in the `Qs` variable inside the script, along with the relative path, `data_path`, to the where the dataframe should be stored. In the folder structured depicted above `data_path = ./data/`. The final dataframe is stored in `dataframe_path = ./data/dataframes`.

## separate_questionnaires.py
---------------------------------------------------

The script separates a dataframe containing data from several questionnaires into
dataframes for each specific questionnaire after the prepareing-procedure is done.

## get_sub_dataframe.py
-----------------------------------------------------
Makes  a sub-dataframe only containing specified variables, run `get_sub_dataframe.py`. The `./data/variables.hjson` file contains the variable names of several variables from different questionnaires. The variables to include in the sub-dataframe must be listed in this hjson file. Inside the script, the variables of interest are listed in `variable_keys`, where the keys corresponds with the hjson file.

The script excludes women that have more than one pregnancy registered in the dataset, by identifying the corresponding mother ids. The mother ids are saved in a binary numpy file and stored in `./data/ids/`.

The dataframe is split into a train and test partition, and gives you the choice to make sure all NaNs are correctly encoded, referred to as cleaning, and to impute the data.
