import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os
import sys
import hjson
from time import time
import pdb

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .print import print_info

class Imputer(object):
    """
    Klasse for å imputere numeriske verdier i en dataframe. Krever at dataframen kun har 
    numeriske verdier i alle kolonner.

    Args:
        dataframe: pandas dataframe. All columns must have a numerical dtype.
    """

    def __init__(self, dataframe):
        self.df = dataframe
      
        # Verify that all columns are infact numeric

        no_numeric = []
        types = []
        for col in self.df.columns:
            try:
                assert is_numeric_dtype(self.df[col])
            except AssertionError:
                if self.df[col].dtype == "O":
                    
                    self.df[col] = self.df[col].astype(float)

                elif self.df[col].dtype == "str":
                    
                    self.df[col] = self.df[col].astype(float)
                   
                else:
                    no_numeric.append(col)
                    types.append(self.df[col].dtype)

        if not len(no_numeric) == 0:
            raise ValueError(f"Column(s) {no_numeric} are not numeric, but {types}")
        

        self._debug = False
    
    def __call__(self):
        
        # Empty dataframe to hold the complete dataframe after the imputations
        self.imputed_df = pd.DataFrame()
        

        # Identify which columns are continiuous and binary
        self._identify_variable_types()
        

        # Account for different scenarios
        if len(self.cont_var) > 0 and len(self.binary_var) > 0:

            self._impute_continiuous()

            self._save_tmp()

            self._impute_binary()

            self._delete_tmp()
        
        elif len(self.cont_var) > 0 and len(self.binary_var) == 0:

            self._impute_continiuous()

        else:
            raise ValueError("No numerical variables identified in the dataframe")


    def _identify_variable_types(self):
        """
        The function identifies continiuos and binary variables in a dataframe.
        Requires the dataframe to only contain numerical values.
        """

        # Identify binary and numerical values
        self.binary_var = []
        self.cont_var = []

        for col in self.df.columns:
            not_nan = np.logical_not(np.isnan(pd.unique(self.df[col]))).astype("int")
            unique_items = pd.unique(self.df[col])
            num_unique = len(unique_items)

            max_val = np.max(self.df[col])

            if num_unique == 2 and max_val == 1:
                self.binary_var.append(col)
            elif num_unique == 3 and max_val == 1 and -1 in unique_items:
                self.binary_var.append(col)
            else:
                self.cont_var.append(col)


    def _impute_continiuous(self):
        """
        Imputes continiuous variables with Multiple Imputation with Chained Equations (MICE) using a 
        decision tree regressor.
        """

        print_info("Imputing continiuous variables")
       
        # Decision Tree Regressor
        tree_reg = DecisionTreeRegressor()
        
        # Iterative Imputer object
        imputer = IterativeImputer(estimator=tree_reg, missing_values=np.nan, max_iter=5, verbose=2, random_state=0)
        
        if "PREG_ID_2601" in self.cont_var:
            self.cont_var.remove("PREG_ID_2601")
            
        # Perform imputations
        imputed_cont = imputer.fit_transform(self.df[self.cont_var])

        pdb.set_trace()
        # Place imputed data in a list of dicts. This is for easier add the data to imputed_df
        imputed_dict = [
                        {self.cont_var[j]: imputed_cont[i,j] 
                        for j in range(len(self.cont_var))} 
                        for i in range(imputed_cont.shape[0])
                        ]

        self.imputed_df = self.imputed_df.append(imputed_dict, ignore_index=True)


    def _impute_binary(self):
        """
        Fits a logistic regressor on the already imputed dataset of continiuous variables, and predict the missing binary values in each column. When a binary variable has been imputed it is added to the 
        imputed dataframe and used in the fit of the next logistic regressor.

        """        

        print_info("Imputing binary variables")

        # All indices present
        idx_list = [i for i in range(self.df.shape[0])]

        if self._debug:
            self.imputed_binary_data = []
            self.complete_binary_data = []


        count = 1
        for bin_var in self.binary_var:
            
            print(f"Imputing binary variable {count}/{len(self.binary_var)}", end="\r")
           

            # Inserting the unimputed data in the imputed dataframe
            self.imputed_df[bin_var] = self.df.loc[:,bin_var].to_numpy()

            
            # Identifying the row indices to the NaN values
            nan_rows = np.where(self.imputed_df[bin_var].isnull() == True)[0]

            if len(nan_rows) == 0:
                continue

            # Preparing test data for the logistic regressor
            X_test = self.imputed_df.iloc[nan_rows, :-1]


            # Rows with complete data
            complete_rows = self.imputed_df[bin_var].dropna().index


            # Training data
            X_train = self.imputed_df.iloc[complete_rows, :-1]
            y_train = self.imputed_df.iloc[complete_rows, -1]


           
            # Scaling data
            X_train = StandardScaler().fit_transform(X_train)
            X_test = StandardScaler().fit_transform(X_test)
        

            # Fitting classifier
            logistic = LogisticRegression()
            imputer = logistic.fit(X_train, y_train)
            imputed_binary = imputer.predict(X_test)


            # Inserting imputed values
            self.imputed_df.iloc[nan_rows, -1] = imputed_binary

            
            if self._debug:
                self.imputed_binary_data.append(imputed_binary)
                self.complete_binary_data.append(y_train)

            count += 1


            if self._debug:
                print("Comparing ratios of ones and zeros for the imputed and complete data \n")

                print("Difference: \n")
                for i in range(len(self.binary_var)):
                    ones = complete_binary_data[i].sum()
                    ratio_complete = ones/(np.abs(len(complete_binary_data[i]) - ones))

                    ones = imputed_binary_data[i].sum()
                    ratio_imputed = ones/(np.abs(len(imputed_binary_data[i]) - ones))
                    

                    print(np.abs(ratio_complete - ratio_imputed))
                    

    def _save_tmp(self):
        self.tmp_name = "imputed_cont_df.csv"
        self.imputed_df.to_csv(self.tmp_name, index=False)

    def _delete_tmp(self):
        os.remove(self.tmp_name)

if __name__ == "__main__":

    # Read dict that holds variables of interest
    with open("./../data/variables.hjson", "r") as f:
        variables = hjson.load(f)

    # Keys to the variables that are in the sub dataframe
    variable_keys = [
        "scl",
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
        "ALE",
        "birth"
    ]


    Q_names = ["Q1", "Q3", "Q4", "Q5", "Q6"]

    # Path to experiment folder, i. e. different combinations of questionnaires are one experiment and thus have their own folder
    data_path = "./../experiments/" + "_".join(Q_names) + "/dataframes/"


    var_qs_name = (
                "_".join(variable_keys)
                + "_"
                + "_".join([Q for Q in variables.keys() if Q in Q_names])
                + ".csv"
            )

    # Name of the finale dataframe
    unprocessed_name = "cleaned_sub_dataframe_" + var_qs_name

    imputed_name =  "imputed_tree_sub_dataframe_" + var_qs_name

    df_file = os.path.join(data_path, unprocessed_name)

    df = pd.read_csv(df_file)

    columns = list(df.columns)

    # Imputing

    imp = Imputer(df)
    imp()


    print("Saving imputed dataframe")

    imp.imputed_df.to_csv("./../" + imputed_name, index=False)
