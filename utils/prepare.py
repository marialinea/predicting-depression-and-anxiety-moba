import numpy as np
import pandas as pd
import os
import sys
import hjson
import re
from time import time
import pdb
from tqdm import tqdm


from .find import find_nans
from .dataframe import load_dataframe#, variable_permutation, check_var
from .data import split_data
from .print import print_info
from .imputer import Imputer
from .cleaner import Cleaner

pd.options.mode.chained_assignment = None  # default='warn'


class PrepareData(object):
    """Klasse som preparerer sub_dataframes i wide format, deler inn i test og train, imputerer og evt. preprosesserer

    Args:
        variables: Dict storing variables of interest from the MoBa files for all questionnaires
        variable_keys: List of keys to the variables that are in the sub dataframe
        Q_names: List of the questionnaires in sub dataframe
        target_processing: variable that specifies the preprocessing of the target variable. Can either be "mean" or "sum"
        data_path: Path to the unprepared dataframes
        out_path: Path to store the prepared dataframes
        clean: bool, default True. The data is cleaned w.r.t NaNs. Meaning that all values that are illegal are set to NaN.
        impute: bool, default False. If True, data is imputed.
        aggregate: bool, default False. If True, variables are aggregated 
        save: bool, deafult True. If True, saves dataframes to out_path
        df_name: If dataframe already exists, loads dataframe from data_path
        split: bool, deafult True. If True, splits dataframe into train and test set
    """

    def __init__(self, variables, variable_keys, Q_names, target_processing, data_path, out_path, clean=True, impute=False, aggregate=False, save=True, df_name=None, split=True):

        self.variables = variables
        self.variable_keys = variable_keys
        self.Q_names = Q_names
        self.target_processing = target_processing
        self.data_path = data_path
        self.out_path = out_path
        self.clean = clean
        self.impute = impute
        self.aggregate = aggregate
        self.save = save
        self.split = split

        
        # self.prepare = True        
        self.exists = False

        assert isinstance(clean, bool), f"clean argument is not of type bool but {type(impute)}"
        assert isinstance(impute, bool), f"impute argument is not of type bool but {type(impute)}"
        assert isinstance(aggregate, bool), f"aggregate argument is not of type bool but {type(aggregate)}"
        assert isinstance(save, bool), f"save argument is not of type bool but {type(aggregate)}"

        if self.out_path[-1] != "/":
            self.out_path += "/"


        # Check if out_path exists, if not, make path
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)


        # Identify which variables in self.variable_keys are present in each questionnaire in self.Q_names
        self._identify_present_variables()

        print_info("Loading dataframe \n")


        if df_name is None:
        # Name of the dataframe, containing the item-group name and questionnaires
            self.dataframe_name = (
                "sub_dataframe_"
                + "_".join(self.variable_keys)
                + "_"
                + "_".join([Q for Q in self.variables.keys() if Q in self.Q_names])
                + ".csv"
            )
        else:
            self.dataframe_name = df_name

         # List over variables that are unique, i. e. if they appear in several questionnaires, they are listed only once
        self.unique_variables = []



        self._check_files()
    

        if self.exists is False:
            
            # Load/or construct unpartitioned dataframe
            self.df = load_dataframe(self.dataframe_name, self.data_path)

            if clean is True:
                ########
                pass
                # self._clean()
                #######

            if split is True:
            
                print_info("Splitting data \n")


                self._split_train_test(self.df) 

            else:
                self.train_test_dict = {"data": {"df": self.df, "outfile_path":  self.out_path + self.dataframe_name}}

            

            if impute is True:

         
                self._impute()


            print_info(f"Aggregating scl items into one target, target_processing={self.target_processing} \n")

           
            self._aggregate_target()

            
            if aggregate is True:


                self._aggregate_variables()


            if save is True:


                self._save_dataframes()


    def _check_files(self):
        """
        Function that checks if a possible set of cleaned, imputed and aggregated dataframes with the variables already exists in self.out_path
        """   

        clean = self.clean
        impute = self.impute
        aggregate = self.aggregate 


        # Dict containing dataframe varaibles and outfile name of the two partitions 
        self.train_test_dict = {
            "train": {},
            "test": {}
        } 

        files = os.listdir(self.out_path)

        partitions = {"train": None, "test": None}

        key_words = ["csv"]
        key_words.append("cleaned") if clean is True else None
        key_words.append("imputed") if impute is True else None
        key_words.append("aggregated") if aggregate is True else None

        
        for part in partitions.keys():

            tmp = key_words.copy()
            tmp.append(part)

            for fn in files:

                word_files = re.split("[_.]", fn)


                if all(word in fn for word in tmp) is all(key in fn for key in self.variable_keys) is all(Q in fn for Q in self.Q_names) is True:
                    
                    # Verify that the filename does not include any other words than the one in the list below
                    all_words = tmp + self.variable_keys + self.Q_names + ["sub", "dataframe"]
                    
                    # Remove the words that are present in the filename
                    [word_files.remove(w) for w in all_words]

                    if len(word_files) == 0:
                        self.train_test_dict[part]["df"] = pd.read_csv(self.    out_path + fn)

                        self.train_test_dict[part]["outfile_path"] = self.  out_path + fn

                        partitions[part] = True

                        if part == "train":
                            self.train_name = fn
                        else:
                            self.test_name = fn


        if all(partitions.values()) is True:
            self.exists = True

            print(f"Found an existing file with the key words {key_words.pop(0)}")

            if aggregate is True:
                
                # If loading dataframes, must still identify the unique variables
                # Dict storing all the aggregated unique variable names


                aggregated_names = {
                    "scl"      :  "mean_scl",
                    "LTHofMD"  :  "LTH_of_MD_agg",
                    "RSS"      :  "mean_RSS",
                    "ss"       :  "ss_bin",
                    "edu"      :  "edu_bin",
                    "imm"      :  "imm",
                    "abuse"    :  "abuse_bin",
                    "GSE"      :  "mean_gse",
                    "income"   :  ["income_mother", "income_father"],
                    "anger"    :  "mean_anger",
                    "RSES"     :  "mean_RSES",
                    "SWLS"     :  "mean_SWLS",
                    "ALE"      :  "ALE_bin",
                    "birth"    :  ["birth_w_Q4", "birth_comp_Q4", "birth_exp_Q4"]
                }

                # Iterating over variable present in the loaded dataframe
                for var in self.variable_keys:

                    if var in aggregated_names.keys():

                        # Add the aggregatead names to the unique_variables list
                        if isinstance(aggregated_names[var], str):

                            self.unique_variables.append(aggregated_names[var])

                        elif isinstance(aggregated_names[var], list):

                            self.unique_variables.extend(aggregated_names[var])
    
    def _check_files_exists_OLD(self):
        """
        Function that checks if a possible set of cleaned and imputed dataframes with the variables already exists in self.out_path


        OLD VERSION
        """    

        if self.clean is False and self.impute is False:

            self.train_name = "train_" + self.dataframe_name
            self.test_name = "test_" + self.dataframe_name

            if os.path.exists(self.out_path + self.train_name) and os.path.exists(self.out_path + self.test_name):
                self.prepare = False
                self.exists = True

                # Load dataframes
                self.df_train = load_dataframe(self.train_name, self.out_path)
                self.df_test = load_dataframe(self.test_name, self.out_path)


        elif self.clean is True and self.impute is False:

            self.train_name = "train_cleaned_" + self.dataframe_name
            self.test_name = "test_cleaned_" + self.dataframe_name

            if os.path.exists(self.out_path + self.train_name) and os.path.exists(self.out_path + self.test_name):
                self.prepare = False
                self.exists = True

                # Load dataframes
                self.df_train = load_dataframe(self.train_name, self.out_path)
                self.df_test = load_dataframe(self.test_name, self.out_path)

        elif self.clean is False and self.impute is True:

            self.train_name = "train_imputed_" + self.dataframe_name
            self.test_name = "test_imputed_" + self.dataframe_name

            if os.path.exists(self.out_path + self.train_name) and os.path.exists(self.out_path + self.test_name):
                self.prepare = False
                self.exists = True

                # Load dataframes
                self.df_train = load_dataframe(self.train_name, self.out_path)
                self.df_test = load_dataframe(self.test_name, self.out_path)

        elif self.clean is True and self.impute is True:
            self.train_name = "train_imputed_cleaned_" + self.dataframe_name
            self.test_name = "test_imputed_cleaned_" + self.dataframe_name

            if os.path.exists(self.out_path + self.train_name) and os.path.exists(self.out_path + self.test_name):
                self.prepare = False
                self.exists = True

                # Load dataframes
                self.df_train = load_dataframe(self.train_name, self.out_path)
                self.df_test = load_dataframe(self.test_name, self.out_path)

        if self.prepare is False:

            self.outfile_train = self.out_path + self.train_name
            self.outfile_test = self.out_path + self.test_name

            # Dict containing dataframe varaibles and outfile name of the two partitions 
            self.train_test_dict = {
                "train": {"df": self.df_train, "outfile_path": self.outfile_train},
                "test": {"df": self.df_test, "outfile_path": self.outfile_test}
            } 
     
    def _update_variables_dict(self):
        """
        Function for updating the variables dictionary hjson file
        """

        with open("./data/variables.hjson", "w") as outfile:
            hjson.dump(self.variables, outfile, indent=4)

    def _identify_present_variables(self):
        """
        Function that identifies variable that are present in the different questionnaires.
        All variables in self.variable_keys are not necessarily available for all the questionnaires in Q_names. Place variables that
        are infact present in self.variable_keys
        """
        present_variables = []

        for Q in self.Q_names:
            for var in self.variables[Q].keys():
                if var in self.variable_keys and var not in present_variables:
                    present_variables.append(var)

        self.variable_keys = present_variables

    def _split_train_test(self, df):
        """
        Function for splitting the data into a train and test partition
        """

        # Name of outfile names
        if self.clean is True:
            self.train_name = "train_cleaned_" + self.dataframe_name
            self.test_name = "test_cleaned_" + self.dataframe_name
        else:
            self.train_name = "train_" + self.dataframe_name
            self.test_name = "test_" + self.dataframe_name


        # Complete paths and file names for the preprocessed dicts
        self.outfile_train = self.out_path + self.train_name
        self.outfile_test = self.out_path + self.test_name



        # Partitioning dataframes
        self.df_train, self.df_test = split_data(df)


        # Dict containing dataframe varaibles and outfile name of the two partitions 
        self.train_test_dict = {
            "train": {"df": self.df_train, "outfile_path": self.outfile_train},
            "test": {"df": self.df_test, "outfile_path": self.outfile_test}
        } 

    def _aggregate_target(self):
        """
        Function that aggregates the target variable, scl score, into either a mean scl score or a sum of total scl score for each item in the symptom checklist.
        """

        # List of target preprocessing values
        target_procedures = ["mean", "Mean", "sum", "Sum"]
        if self.target_processing not in target_procedures:
            raise ValueError("Argument 'target_processing' must be either 'mean' or 'sum', not {}".format(target_processing))
        elif self.target_processing == "mean" or target_processing == "Mean":
            self.target_processing = "mean"
        elif self.target_processing == "sum" or target_processing == "Sum":
            self.target_processing = "sum"

        target_function = self._scl_sum if self.target_processing == "sum" else self._mean_scl

        # Preparing the target for train and test paritions separately
        for key in self.train_test_dict.keys():
            #print("Preparing target variables the {} partition".format(key), end="\r")

            # Make the specific dataframe a class variable
            self.df = self.train_test_dict[key]["df"]

            self.variable_key = "scl"

            # Applying aggregation function to target
            target_function()

    def _save_dataframes(self):
        """
        Function that save the dataframes in the partiotion_dict dictionary to the path given in said dictionary.
        """

        if self.exists is False:

            for key in self.train_test_dict:

                if os.path.exists(self.train_test_dict[key]["outfile_path"]) is False:
                 
                    self.train_test_dict[key]["df"].to_csv(self.train_test_dict[key]["outfile_path"], index=False)
        else:
            print(f"Dataframes already exists in {self.out_path}")

            save = input("Still save dataframes (y/[n])? ")
            print("")
            
            if save == "y":
                for key in self.train_test_dict:

                    if os.path.exists(self.train_test_dict[key]["outfile_path"]) is False:

                        self.train_test_dict[key]["df"].to_csv(self.train_test_dict[key]["outfile_path"], index=False)
            else:
                return

    def _clean_OLD_NOT_WORKING(self):

        self.dataframe_name = "cleaned_" + self.dataframe_name
        
        for key in self.train_test_dict.keys():
            print_info("Cleaning data for", end=""); print(f"\x1b[94m {key} \033[0m", end=""); print_info("partition \n")
            

            cleaner = Cleaner(self.train_test_dict[key]["df"], self.variable_keys, self.variables, self.Q_names)
            cleaner()


            self.train_test_dict[key]["df"] = cleaner.df


            self.train_test_dict[key]["outfile_path"] = f"{self.out_path}{key}_{self.dataframe_name}"

    def _clean(self):

        print_info("Cleaning dataframe \n")

        self.dataframe_name = "cleaned_" + self.dataframe_name
        

        cleaner = Cleaner(self.df, self.variable_keys, self.variables, self.Q_names)
        cleaner()


        self.df = cleaner.df
 
    def _impute(self):

        self.dataframe_name = "imputed_" + self.dataframe_name
        self.train_name = "imputed_" + self.train_name
        self.test_name = "imputed_" + self.test_name

        for key in self.train_test_dict.keys():
            print_info("Imputing missing data for", end=""); print(f"\x1b[94m {key} \033[0m", end=""); print_info("partition \n")
            
            imputer = Imputer(self.train_test_dict[key]["df"])
            imputer()

            self.train_test_dict[key]["df"] = imputer.imputed_df


            self.train_test_dict[key]["outfile_path"] = f"{self.out_path}{key}_{self.dataframe_name}"

    def _aggregate_variables(self):
        """
        Function that aggregate items in the same variable-group. The function calls all of the aggregation functions.
        """

        self.dataframe_name = "aggregated_" + self.dataframe_name
        
        if self.split is True:
            self.train_name = "aggregated_" + self.train_name
            self.test_name = "aggregated_" + self.test_name
        else:
            self.train_test_dict["data"]["outfile_path"] =  self.out_path + self.dataframe_name

        # Dict storing all the functions for aggregating for specific variables in the variables.hjson file
        aggregated_functions = {
            "LTHofMD" : self._LTH_of_MD_agg,
            "RSS"     : self._mean_RSS,
            "ss"      : self._social_support_bin,
            "edu"     : self._education_bin,
            "imm"     : self._immigration_status,
            "abuse"   : self._abuse_agg,
            "GSE"     : self._mean_gse,
            "income"  : self._income,
            "anger"   : self._mean_anger,
            "RSES"    : self._mean_RSES,
            "SWLS"    : self._mean_SWLS,
            "ALE"     : self._adverse_life_events_bin,
            "birth"   : self._birth_vars 
        }
        
        t0 = time()

        # Aggregating the variables in the train and test partitions separately
        for key in self.train_test_dict.keys():

            print_info("Aggregating", end=""); print(f"\x1b[94m {key} \033[0m", end=""); print_info("partition \n")


            # Make the specific dataframe a class variable
            self.df = self.train_test_dict[key]["df"]


            # Applying aggregation functions to features
            for var in self.variable_keys:


                if var == "scl":
                    continue
                try:

                    # Making the current variable a class member
                    self.variable_key = var


                    # Calling preprocess functions for specific variable
                    aggregated_functions[var]()


                except KeyError:
                    print(
                        f"Aggregation function not implemented for the variable {var}. \n"
                    )


            # Place the aggregated dataframe in the correct class variable
            self.train_test_dict[key]["df"] = self.df
            # pdb.set_trace()


        # The self.unique_variables will contain duplicates of every variable that is unique because it appends to the list for both the train and test set.
        self.unique_variables = list(set(self.unique_variables))


        print("Aggregation compledet in {:.1f} s".format(time() - t0))

    
    def _scl_sum(self):
        """Aggregate all of the scl variables into one variable additively.  The original scl variables are then removed from the dataframe inplace"""

        print("Adding all of the scl variables into one")

        t0 = time()

        agg_names = ["sum_scl_{}".format(i) for i in self.Q_names]

        # Counts how many columns parsed
        counter = 0

        # Iterating through the variables dict
        for Q, nested_dict in self.variables.items():

            # Only including the variables from the specified questionnaires
            if Q in self.Q_names:

                # Iterating through the variables for each questionnaire
                for key, values in nested_dict.items():

                    # Extracting the columns of interest
                    if key in self.variable_key:

                        # Index of new column containing the means
                        new_index = self.df.columns.get_loc(values[-1]) + 1
                        
                        # Inserting new column. Scaling the answer from Q1 since here SCL-5 is used as opposed to SCL-8
                        if Q == "Q1":
                            self.df.insert(
                                new_index, agg_names[counter], self.df[values].sum(axis=1)*1.6
                            )
                        else:
                            self.df.insert(
                                new_index, agg_names[counter], self.df[values].sum(axis=1)
                            )
                        
                       
                        # Removing variable columns
                        self.df.drop(labels=values, axis=1, inplace=True)

                        counter += 1
        
        # Updating variables dict
        counter = 0
        for Qs, nested_dict in self.variables.items():
            if Qs in self.Q_names:
                self.variables[Qs][agg_names[counter]] = agg_names[counter]
                counter += 1

        self._update_variables_dict()


        #  Setting values that are above 32 to NaN
        self.df[agg_names] = self.df[agg_names][self.df[agg_names] <= 32]

        # ---- Indentifying rows containing NaN values for the dependent variables ----

        nan_indices = find_nans(self.df, self.variables, agg_names)

        # Removing the rows containing NaN values
        self.df.drop(index=nan_indices, axis=0, inplace=True)

        # Appending the unique variable name to the unique_variables list
        tmp = agg_names[0].split("_")
        unique_name = "_".join((tmp[0], tmp[1]))
        self.unique_variables.append(unique_name)

        print("Done in {:.1f}s \n".format(time() - t0))

    def _mean_scl(self):
        """Calculates means for the scl variables in each questionnaire and stores them in new columns. The original scl variables are then removed from the dataframe inplace."""

        # Column names for the mean variable
        mean_names = ["mean_scl_{}".format(i) for i in self.Q_names[:-1]]

        # Calculating and adding the means
        self._get_mean_columns(mean_names)

        #  Setting mean values that are above 4 to NaN
        # self.df[mean_names] = self.df[mean_names][self.df[mean_names] <= 4]

        # ---- Indentifying rows containing NaN values for the dependent variables ----

        # nan_indices = find_nans(self.df, self.variables, mean_names)

        # Removing the rows containing NaN values
        # self.df.drop(index=nan_indices, axis=0, inplace=True)

        # Appending the unique variable name to the unique_variables list
        tmp = mean_names[0].split("_")
        unique_name = "_".join((tmp[0], tmp[1]))
        self.unique_variables.append(unique_name)

    def _get_mean_columns(self, new_names):
        """Calculates row wise mean of selected columns and place the means in a new column

        Args:
            new_names: list holding the names of the new columns in the format ['newname_Q1', 'newname_Q2']

        Return:
            Alters the dataframe in-place, no return statement
        """
        # Identify which questonnaires are represented in the columns
        Qs_present = [q.split("_")[-1] for q in new_names]
  
        # Counts how many columns parsed
        counter = 0

        # Iterating through the variables dict
        for Qs, nested_dict in self.variables.items():

            # Only including the variables from the specified questionnaires
            if Qs in Qs_present:

                # Iterating through the variables for each questionnaire
                for key, values in nested_dict.items():

                    # Extracting the columns of interest
                    if key in self.variable_key:

                        # Index of new column containing the means
                        new_index = self.df.columns.get_loc(values[-1]) + 1

                        # Inserting new column
                        self.df.insert(
                            new_index, new_names[counter], self.df[values].mean(axis=1)
                        )

                        # Removing variable columns
                        self.df.drop(labels=values, axis=1, inplace=True)

                        counter += 1

        # Updating variables dict
        counter = 0
        for Qs, nested_dict in self.variables.items():
            if Qs in Qs_present:
                self.variables[Qs][new_names[counter]] = new_names[counter]
                counter += 1

        self._update_variables_dict()

    def _LTH_of_MD_agg(self):
        """
        Aggregating the variables for lifetime hisotry of majo depression (LTH_of_MD) variable
        """

        print("Aggregating LTH_of_MD variables into one dichotomous variable")


        # Column name of the aggregated variable
        agg_variable = "LTH_of_MD_agg"

        LTH_key = self.variables["Q1"][self.variable_key]

        # Summing over the two first LTH_of_MD variables
        criteria_1_2 = self.df[LTH_key[:-1]].sum(axis=1)

        # The last LTH_of_MD variable
        criterion_3 = self.df[LTH_key[-1]]

        # Creating new column
        self.df[agg_variable] = np.zeros(self.df.shape[0])

        if self.clean is True:
            # Ensuring that the three DSM-III criteria are fulfilled
            counter = 0
            for crit_1_2, crit_3 in zip(criteria_1_2, criterion_3):

                # Answered 'yes' to both first and last questions in the LTH_of_MD instrument and 'no' to external reasons behind the emotions
                if crit_1_2 == 2 and (crit_1_2 + crit_3) == 2:
                    self.df[agg_variable].iloc[counter] = 1
                else:
                    self.df[agg_variable].iloc[counter] = 0
                counter += 1
        else:
            # Ensuring that the three DSM-III criteria are fulfilled
            counter = 0
            for crit_1_2, crit_3 in zip(criteria_1_2, criterion_3):

                # Answered 'yes' to both first and last questions in the LTH_of_MD instrument and 'no' to external reasons behind the emotions
                if crit_1_2 == 4 and (crit_1_2 + crit_3) == 5:
                    self.df[agg_variable].iloc[counter] = 1
                else:
                    self.df[agg_variable].iloc[counter] = 0
                counter += 1

        # Updating dataframe by removing the LTH_of_MD columns
        self.df.drop(labels=LTH_key, axis=1, inplace=True)

        # Updating variable dict
        self.variables["Q1"][agg_variable] = agg_variable

        self._update_variables_dict()

        # Appending the variable to the unique_variables list
        self.unique_variables.append(agg_variable)

    def _mean_RSS(self):
        """
        Calculates means for the RSS variables in each questionnaire and stores them in new columns. The original RSS variables are then removed from the dataframe inplace.
        """

        print("Calculating means for the RSS variables in each questionnaire")
    

        # Column names for the mean variable
        mean_names = ["mean_RSS_{}".format(i) for i in self.Q_names]
       
        # Calculating and adding the means
        self._get_mean_columns(mean_names)

        # All values above 6 are set to NaN, since 6 is the highest value possible to obtain
        self.df[mean_names] = self.df[mean_names][self.df[mean_names] <= 6]

        # Appending the unique variable name to the unique_variables list
        tmp = mean_names[0].split("_")
        unique_name = "_".join((tmp[0], tmp[1]))
        self.unique_variables.append(unique_name)

    def _social_support_bin(self):
        """Turning social support variable into dichotomous variable

        The maximum value of this instrument is 3, so the function set all values larger than 3 to NaN, before turning it into a dichotomous variable.

        """

        print(
            "Transforming variable concerning social support into dichotomous variable"
        )
        
        # Column name of the binary variable
        binary_variable = "ss_bin"

        # Key to the social support question in self.variables
        ss_key = self.variables["Q1"][self.variable_key]

        # All values above 3 are set to NaN, since 3 is the highest value for the instrument
        self.df[ss_key] = self.df[ss_key][self.df[ss_key] <= 3]

        # Creating new binary column
        self.df[binary_variable] = (
            self.df[self.variables["Q1"][self.variable_key]] > 1
        ).astype("int")

        # Updating dataframe by removing the social support column
        self.df.drop(labels=ss_key, axis=1, inplace=True)

        # Updating variable dict
        self.variables["Q1"][binary_variable] = binary_variable

        self._update_variables_dict()

        # Appending the variable to the unique_variables list
        self.unique_variables.append(binary_variable)

    def _education_bin(self):
        """Transforming the education variables into a dichotomous variable. Finished or started higher education is coded as 1, if not the variable is coded as 0.

        The maximum value of this instrument is 6, so the function set all values larger than 6 to NaN, before turning it into a dichotomous variable.
        """

        print("Transforming variables concerning education into dichotomous variable")
    
        # Column name of the binary variable
        binary_variable = "edu_bin"

        # Key to the educational questions in self.variables
        edu_key = self.variables["Q1"][self.variable_key]

        # All values above 6 are set to NaN, since 6 is the highest value for the instrument
        self.df[edu_key] = self.df[edu_key][self.df[edu_key] <= 6]

        # Binary array storing information about completed education
        completed = (self.df[self.variables["Q1"][self.variable_key][0]] > 4).astype(
            "int"
        )

        # Binary array storing information about started education
        started = (self.df[self.variables["Q1"][self.variable_key][1]] > 4).astype(
            "int"
        )

        # If the sum of the two arrays above are greater than 0, the participant has either completed or started higher education
        self.df[binary_variable] = (completed + started > 0).astype("int")

        self._update_variables_dict()

        # Updating dataframe by removing the educational columns
        self.df.drop(labels=edu_key, axis=1, inplace=True)

        # Appending the variable to the unique_variables list
        self.unique_variables.append(binary_variable)

    def _immigration_status(self):
        """To determine immigration status a variable concerning mother tounge language is used. This variable is either coded 1 or 2, 1 for Norwegian as native language, 2 for others. The functions changes the coding from 1/2 --> 0/1"""

        print("Transforming variables concerning immigration status")
    
        # Key to the language question in self.variables
        immigration_key = self.variables["Q1"][self.variable_key]


        if not self.clean is True:
            # All values above 1 are set to NaN, since 1 is the highest value for the instrument
            self.df[immigration_key] = self.df[immigration_key][
                self.df[immigration_key] <= 1
            ]

            # Change coding from 1/2 --> 0/1
            self.df[immigration_key] = self.df[immigration_key].transform(
                np.vectorize(lambda x: x - 1)
            )

        # Renaming immigration column name from immigration_key to self.variable_key
        self.df.rename(columns={immigration_key[0]: self.variable_key}, inplace=True)

        # Appending the variable to the unique_variables list
        self.unique_variables.append(self.variable_key)

    def _abuse_agg(self):
        """Turning abuse variables into a dichotomous variable

        The function expects 12 abuse questions in total, where pairs of 3 belongs to one type of abuse assessment. First question in the trio corresponds do no {emotional, physical, sexual} abuse, the remaining determines when it happend, childhood or adulthood.
        """

        print("Transforming variables concerning abuse into dichotomous variable")
       
        # Column name of the new binary variable
        binary_variable = "abuse_bin"

        # Key to the abuse questions in self.variables
        abuse_key = self.variables["Q3"][self.variable_key]

        # Defining new binary variable
        self.df[binary_variable] = np.zeros(self.df.shape[0])

        # List over questions that specifies 'no abuse'
        no_abuse_qs = [abuse_key[i] for i in range(0, 12, 3)]

        # All NaN values in the abuse question columns are set to 0
        self.df[abuse_key] = self.df[abuse_key].fillna(0)

        # Iterating through every pregnancy
        for i in tqdm(
            range(self.df.shape[0]), desc="Iterating through rows", leave=False
        ):

            # If answered 'yes' to all of the 'no abuse' questions, set binary variable to zero
            if self.df[no_abuse_qs].iloc[i].sum() == 4:
                self.df[binary_variable].iloc[i] = 0
            else:
                # Iterating thorugh the different abuse questions
                for j in range(0, 12, 3):
                    q1 = abuse_key[j]
                    q2 = abuse_key[j + 1]
                    q3 = abuse_key[j + 2]

                    # If answered no to the abuse question
                    if self.df[q1].iloc[i] and self.df[[q2, q3]].iloc[i].sum() == 0:
                        self.df[binary_variable].iloc[i] = 0

                    # If answered no and not yes to any questions
                    elif (
                        not self.df[q1].iloc[i] and self.df[[q2, q3]].iloc[i].sum() == 0
                    ):
                        # self.df[binary_variable].iloc[i] = np.nan
                        self.df[binary_variable].iloc[i] = 0

                    # If answered yes to any of the last two questions
                    elif (
                        not self.df[q1].iloc[i] and self.df[[q2, q3]].iloc[i].sum() > 0
                    ):
                        self.df[binary_variable].iloc[i] = 1
                        break

        # Updating dataframe by removing the social support column
        self.df.drop(labels=abuse_key, axis=1, inplace=True)

        # Updating variable dict
        self.variables["Q3"][binary_variable] = binary_variable

        self._update_variables_dict()

        # Appending the variable to the unique_variables list
        self.unique_variables.append(binary_variable)

    def _income(self):
        """Function that verifies that the income variables have legal values, i. e. for first question no value is < 8, and for the second no value is < 9."""

        print("Verifying income variables")
    

        # Key to the income questions in self.variables
        income_key = self.variables["Q1"][self.variable_key]

        # List over maximum legal values for the two income variables
        max_values = [7, 8]

        #  Identifying illegal values
        for i in range(len(income_key)):

            if self.df[income_key[i]].max() > max_values[i]:

                #  Setting values that are above 7 to NaN
                self.df[income_key[i]] = self.df[income_key[i]][
                    self.df[income_key[i]] <= max_values[i]
                ]

        # Renaming income column names from 'AA1315' to 'income_mother' and 'AA1316' to 'income_father'
        new_names = {"AA1315": "income_mother", "AA1316": "income_father"}
        self.df.rename(columns=new_names, inplace=True)

        # Appending the unique variable names to the unique_variables list
        self.unique_variables.extend(list(new_names.values()))

    def _mean_gse(self):
        """Calculates means for the gse variables in each questionnaire and stores them in new columns. The original gse variables are then removed from the dataframe inplace."""

        print("Calculating means for the gse variables in each questionnaire")

        # Column names for the mean variable
        mean_names = ["mean_gse_Q3"] #, "mean_gse_Q5"]

        # Calculating and adding the means
        self._get_mean_columns(mean_names)

        # Remove the mean_anger_Q3 key from the variables dict
        self.variables["Q3"].pop(mean_names[0])

        #  Setting mean values that are above 4 to NaN
        self.df[mean_names] = self.df[mean_names][self.df[mean_names] <= 4]

        # Rename mean_gse_Q3 column to simply mean_gse
        new_name = {mean_names[0]: "mean_gse"}
        self.df.rename(columns=new_name, inplace=True)

        # Updating variable dict
        self.variables["Q3"][list(new_name.values())[0]] = list(new_name.values())[0]

        self._update_variables_dict()

        # Appending the unique variable name to the unique_variables list
        self.unique_variables.append(list(new_name.values())[0])


        # The block of code below is for two values of GSE at different times
        # #  Setting mean values that are above 4 to NaN
        # self.df[mean_names] = self.df[mean_names][self.df[mean_names] <= 4]

        # # Updating variable dict
        # self.variables["Q3"][mean_names[0]] = mean_names[0]
        # self.variables["Q5"][mean_names[1]] = mean_names[1]

        # self._update_variables_dict()

        # # Appending the unique variable name to the unique_variables list
        # tmp = mean_names[0].split("_")
        # unique_name = "_".join((tmp[0], tmp[1]))
        # self.unique_variables.append(unique_name)

    def _mean_anger(self):
        """Calculates the mean for the anger variables in Q3 and stores it in a new column. The original anger variables are then removed from the dataframe inplace."""

        print("Calculating the mean for the anger variables in Q3")

        # Column names for the mean variable
        mean_names = ["mean_anger_Q3"]

        # Calculating and adding the means
        self._get_mean_columns(mean_names)

        # Remove the mean_anger_Q3 key from the variables dict
        self.variables["Q3"].pop(mean_names[0])

        #  Setting mean values that are above 5 to NaN
        self.df[mean_names] = self.df[mean_names][self.df[mean_names] <= 5]

        # Rename mean_anger_Q3 column to simply mean_anger
        new_name = {mean_names[0]: "mean_anger"}
        self.df.rename(columns=new_name, inplace=True)

        # Updating variable dict
        self.variables["Q3"][list(new_name.values())[0]] = list(new_name.values())[0]

        self._update_variables_dict()

        # Appending the unique variable name to the unique_variables list
        self.unique_variables.append(list(new_name.values())[0])

    def _mean_RSES(self):
        """Calculates the mean for the Rosenberg Self-Esteem Scale (RSES) variables in Q3 and stores it in a new column. The original RSES variables are then removed from the dataframe inplace."""

        print("Calculating the mean for the RSES variables in Q3")

        # Column names for the mean variable
        mean_names = ["mean_RSES_Q3"]

        # Calculating and adding the means
        self._get_mean_columns(mean_names)

        # Remove the mean_anger_Q3 key from the variables dict
        self.variables["Q3"].pop(mean_names[0])

        #  Setting mean values that are above 4 to NaN
        self.df[mean_names] = self.df[mean_names][self.df[mean_names] <= 4]

        # Rename mean_RSES_Q3 column to simply mean_RSES
        new_name = {mean_names[0]: "mean_RSES"}
        self.df.rename(columns=new_name, inplace=True)

        # Updating variable dict
        self.variables["Q3"][list(new_name.values())[0]] = list(new_name.values())[0]

        self._update_variables_dict()

        # Appending the unique variable name to the unique_variables list
        self.unique_variables.append(list(new_name.values())[0])

    def _mean_SWLS(self):
        """Calculates the mean for the Satisfaction With Life Scale (SWLS) variables in Q1 and stores it in a new column. The original SWLS variables are then removed from the dataframe inplace."""

        print("Calculating the mean for the SWLS variables")

        # Column names for the mean variable
        mean_names = ["mean_SWLS_{}".format(i) for i in self.Q_names]
        
        # Calculating and adding the means
        self._get_mean_columns(mean_names)

        # All values above 6 are set to NaN, since 6 is the highest value possible to obtain
        self.df[mean_names] = self.df[mean_names][self.df[mean_names] <= 6]

        # Appending the unique variable name to the unique_variables list
        tmp = mean_names[0].split("_")
        unique_name = "_".join((tmp[0], tmp[1]))
        self.unique_variables.append(unique_name)

        self._update_variables_dict()

    def _adverse_life_events_bin(self):
        """Adverse Life Events (ALE) are encoded as four binary values, if answered yes to any of the nine questions related to ALE in Q3-Q6,
        the variable for the specific questionnaire is coded as 1. If answered no, coded as 0.

        For Q1, all mothers have a ALE_bin_Q1 variable that is set to 0.
        """

        print("Aggregating variables concerning adverse life events")

        # Column names for the binary variables
        bin_names = ["ALE_bin_{}".format(i) for i in self.Q_names[1:]]

        # # Hard coding in the value for Q1
        # if "Q1" in self.Q_names:
        #     self.df["ALE_bin_Q1"] = 0
        #     self.variables["Q1"]["ALE_bin_Q1"] = "ALE_bin_Q1"
        
        for i, Q in enumerate(self.Q_names):
            if self.variable_key in self.variables[Q].keys():

                # Keys to the ALE variables in each questionnaire
                ALE_keys = self.variables[Q][self.variable_key]

                if not self.clean is True:
                    # Change coding from 1/2 --> 0/1
                    self.df[ALE_keys] = self.df[ALE_keys].transform(
                        np.vectorize(lambda x: x - 1)
                    )

                # Creating new binary variable
                self.df[bin_names[i-1]] = (self.df[ALE_keys].sum(axis=1) > 0).astype(
                    "int"
                )

                # Updating variables dict
                self.variables[Q][bin_names[i-1]] = bin_names[i-1]

                # Removing variable columns
                self.df.drop(labels=ALE_keys, axis=1, inplace=True)

        self._update_variables_dict()

        # Appending the binary variable to the unique_variables list
        tmp = bin_names[0].split("_")
        unique_name = "_".join((tmp[0], tmp[1]))
        self.unique_variables.append(unique_name)

    def _birth_vars(self):
        """
        Function to rename the three birth items. They are kept as they are.
        """

        birth_items = self.variables["Q4"][self.variable_key]

        new_names = {birth_items[0]: "birth_w_Q4", birth_items[1]: "birth_comp_Q4", birth_items[2]: "birth_exp_Q4"}
        self.df.rename(columns=new_names, inplace=True)

        self.unique_variables.extend(list(new_names.values()))