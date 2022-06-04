## GetMotherPregIds.py
----------------------------------------------------

The script identifies the pregnancy ids and connects them with the mother id in  a dictionary.
This means that a single mother id can be connected to several pregnancy ids. The data is stored
in the json format in a dictionary on the form
{'motherID_1': [pregnancy_ID_1, ..., pregnancy_ID_n], 'mother_ID_2': [pregnancy_ID_1, ..., pregnancy_ID_m], ...}


## separate_questionnaires.py
---------------------------------------------------

The script separates a dataframe containing data from several questionnaires into
dataframes for each specific questionnaire after the prepareing-procedure is done.

