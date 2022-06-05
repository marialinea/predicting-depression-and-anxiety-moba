# Utilities
This folder contains various utility functions that have contributed into processing data and making predictions.

## cleaner.py

Contains a class that makes sure that all values in the base dataset are correctly encoded.


## configure.py

Class that sets up and prepare all of the underlying structures and data for the experiments. All of the different model classes in the `main_XXX.py` scripts inherent from this class.

## data.py

Contain various functions that prepare and handles the data.

## dataframe.py

Functions for loading, merging, finding specific columns and converting wide data into long data.

## find.py

Functions for finding nan, mother ids and unique pregnancy ids.

## imputer.py

Contains a class that imputes the data using MICE with decision trees.

## prepare.py

This class incorporates the imputer and cleaner classes to make one class for the whole preprocessing procedure.

## print.py

Holds simple print functions that print different colours.
