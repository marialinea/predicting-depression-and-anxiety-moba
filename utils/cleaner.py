import pandas as pd
import numpy as np
import hjson
import json
import sys
import os
import time
import operator
import pdb
from tqdm import tqdm

from .print import print_info



class Cleaner(object):
    """
    Klasse for å sørge for at alle NaN verdier er korrekt. I mange tilfeller er det NaN verdier som burde
    vært kodet som "Nei". 

    Args:
        dataframe: pandas dataframe, a sub_dataframe with the variables corresponding with the variable_keys
        variables: Dict storing variables of interest from the MoBa files for all questionnaires
        variable_keys: List of keys to the variables that are in the sub dataframe
        Q_names: List of the questionnaires in sub dataframe
        verbose: variable for printing feedback. Accepted values are 0 and 1.
    """

    def __init__(self, dataframe, variable_keys, variables, Q_names, verbose=0):

        self.df = dataframe
        self.variable_keys = variable_keys
        self.variables = variables
        self.Q_names = Q_names
        self.verbose = verbose

        if verbose not in [0,1]:
            raise ValueError(f"verbose = {verbose}, accepted values are 0 or 1")

        # Dict to store all items for one variable for all questionnaires in Q_names
        self.items = {}
        self._fill_items_dict()

        # Total number of variables that have been changed from an illegal value to NaN
        self.total_count = 0

        # Dict containing legal values for the different variables. 
        # If a single number for min/max val, this is valid for all items for that variable.
        # If a list, the length of the list must be equally long as the number of items for that variable.
        # E. g. there are three birth items. 

        self.cleanup_dict = {
            "scl"                 :  {"min": 1, "max": 4},
            "LTHofMD"             :  {"min": None, "max": None, "special_cleanup": self._clean_LTHofMD},
            "RSS"                 :  {"min": 1, "max": 6, "special_cleanup": self._clean_rss},
            "ss"                  :  {"min": 1, "max": 3},
            "edu"                 :  {"min": 1, "max": 6, "special_cleanup": self._clean_edu},
            "imm"                 :  {"min": 1, "max": 2, "special_cleanup": self._clean_imm},
            "abuse"               :  {"min": 1, "max": 1, "special_cleanup": self._clean_abuse},
            "income"              :  {"min": [1,1], "max": [7,8]},
            "GSE"                 :  {"min": 1, "max": 4},
            "anger"               :  {"min": 1, "max": 5},
            "RSES"                :  {"min": 1, "max": 4},
            "SWLS"                :  {"min": 1, "max": 7},
            "ALE"                 :  {"min": 1, "max": 2, "special_cleanup": self._clean_ALE},
            "birth"               :  {"min": [0,1,1], "max": [20_000,2,5], "special_cleanup": self._clean_birth},
            "SocialSupport"       :  {"min": [1,1,1], "max": [3,3,5]},
            "Work_Q3"             :  {"min": None, "max": None, "special_cleanup": self._clean_workQ3},
            "Work_Q6"             :  {"min": None, "max": None, "special_cleanup": self._clean_workQ6},
            "SickLeave_Q3"        :  {"min": None, "max": None, "special_cleanup": self._clean_sickleaveQ3},
            "SickLeave_Q4"        :  {"min": None, "max": None, "special_cleanup": self._clean_sickleaveQ4},
            "Drugs"               :  {"min": None, "max": None, "special_cleanup": self._clean_drugs},
            "Alcohol"             :  {"min": None, "max": None, "special_cleanup": self._clean_alcohol},
            "Emotion"             :  {"min": 1, "max": 5},
            "AdverseLifeEvents"   :  {"min": [1,1], "max": [2,3], "special_cleanup": self._clean_adverselifeevents},
            "Assault_Q3"          :  {"min": None, "max": None, "special_cleanup": self._clean_assaultQ3},
            "Assault_Q6"          :  {"min": 1, "max": 2, "special_cleanup": self._clean_assaultQ6},
            "Birth"               :  {"min": None, "max": None, "special_cleanup": self._clean_Birth},
            "ChildDevelopment_Q4" :  {"min": 1, "max": 4},
            "ChildDevelopment_Q5" :  {"min": 1, "max": 3},
            "ChildDevelopment_Q6" :  {"min": 1, "max": 3},
            "ChildBehaviour"      :  {"min": 1, "max": 3},
            "ChildManner"         :  {"min": 1, "max": 3},
            "ChildMood"           :  {"min": 1, "max": 7},
            "ChildTemperament"    :  {"min": 1, "max": 5},
            "AdultADHD"           :  {"min": 1, "max": 5},
            "Communication"       :  {"min": None, "max": None, "special_cleanup": self._clean_commnication},
            "EPDS"                :  {"min": 1, "max": 4},
            "CivilStatus"         :  {"min": 1, "max": 6},
            "WHOQOL"              :  {"min": 1, "max": 5},
            "SocialSkills"        :  {"min": 1, "max": 3},
            "PLOC"                :  {"min": 1, "max": 5},
            "SocialCommunication" :  {"min": 1, "max": 2, "special_cleanup": self._clean_socialcommunication},
            "Finance"             :  {"min": [1,1], "max": [3,4]},
            "ChildCare"           :  {"min": None, "max": None, "special_cleanup": self._clean_childcare},
            "ChildLengthWeight"   :  {"min": None, "max": None, "special_cleanup": self._clean_childlengthweight},
            "LivingSituation"     :  {"min": None, "max": None, "special_cleanup": self._clean_livingsituation},
            "LivingWithFather"    :  {"min": None, "max": None, "special_cleanup": self._clean_livingwithfather},
            "LivingEnvironment"   :  {"min": None, "max": None, "special_cleanup": self._clean_livingenvironment},
            "ParentalLeave"       :  {"min": None, "max": None, "special_cleanup": self._clean_parentalleave},
            "TimeOutside"         :  {"min": None, "max": None, "special_cleanup": self._clean_timeoutside},
            "PregnantNow"         :  {"min": None, "max": None, "special_cleanup": self._clean_pregnantnow},
            "Autism"              :  {"min": None, "max": None, "special_cleanup": self._clean_autism},
            "Daycare"             :  {"min": None, "max": None, "special_cleanup": self._clean_daycare},
            "WalkUnaided"         :  {"min": None, "max": None, "special_cleanup": self._clean_walkunaided},
            "Smoking_Q4"          :  {"min": None, "max": None, "special_cleanup": self._clean_smokingQ4},
            "Smoking_Q5"          :  {"min": [1,0,1,0,1,1,1], "max": [3,99,3,99,7,6,6], "special_cleanup": self._clean_smokingQ5},
            "Smoking_Q6"          :  {"min": None, "max": None, "special_cleanup": self._clean_smokingQ6},
            "Lifting"             :  {"min": None, "max": None, "special_cleanup": self._clean_lifting}
        }


    def __call__(self):

        self.operators = {"min": operator.lt, "max": operator.gt}
        operators = self.operators

        for var in self.variable_keys:
            
            try:
                # Allowed range of numerical values
                min_val = self.cleanup_dict[var]["min"]
                max_val = self.cleanup_dict[var]["max"]
                

                min_max = [min_val, max_val]

                # All items for the specific variable
                variable_items = self.items[var]
                

                # Identifying the min and max values for the items in the dataframe
                min_max_var = self.df[variable_items].describe().loc[["min","max"]]
            
                if all([isinstance(min_val, (int, float)), isinstance(max_val, (int, float))]):

                    for i, op in enumerate(operators.keys()):
                        
                    
                        if np.any(operators[op](min_max_var.values[i], min_max[i])):
            
                            # Find columns containing values that are lower/higher than min_val/max_val
                            columns = operators[op](self.df.loc[:,variable_items].describe().loc[op], min_max[i])
                            

                            min_col = np.where(columns==True)[0]


                            col_names = self.df.loc[:, variable_items].columns[min_col]


                            self._set_values_nan(var, col_names, extremum=op)


                elif all([isinstance(min_val, list), isinstance(max_val, list)]):
                    
                    if len(min_val) != len(max_val):
                        raise ValueError("Different lengths for the minimum and maximum list")

                    min_max_var = list(min_max_var.loc["min"].values) + list(min_max_var.loc["max"].values)
                    min_max_list = min_val + max_val
                    min_max_string = ["min" for i in range(len(min_val))] + ["max" for i in range(len(max_val))]
                    
            
                    for i in range(0, len(variable_items), len(min_val)):
                        current_items = variable_items[i:i+len(min_val)]
                        current_items += current_items
                        current_min_max_var = min_max_var[i:i+len(min_val)] + \
                                            min_max_var[i+len(variable_items):i+len(variable_items)+len(min_val)]

                        for op, var_val, allowed_val, item in zip(min_max_string, current_min_max_var, min_max_list, current_items):
                            
                            if np.any(operators[op](var_val, allowed_val)):
                
                    
                                col_name = item


                                self._set_values_nan(var, col_name, extremum=op, ext_val=allowed_val)


                elif all([isinstance(min_val, type(None)), isinstance(max_val, type(None))]):
                    pass

                # min_max_var = self.df[variable_items].describe().loc[["min","max"]]
                
                if "special_cleanup" in self.cleanup_dict[var].keys():
                
                    self.cleanup_dict[var]["special_cleanup"](var)
                

            except KeyError:
                raise NotImplementedError(f"No cleaning procedure implemented for {var}")

        #self.summary()


    def _fill_items_dict(self):
        """
        Helper function to simply identify all items related to one variable and store them in a dictionary
        """
        for key in self.variable_keys:
            self.items[key] = []

        for Q in self.Q_names:
            for var in self.variable_keys:
                try:
                    self.items[var].extend(self.variables[Q][var])
                except KeyError:
                    continue


    def _set_values_nan(self, variable, columns, extremum="min", ext_val=None):
        """
        Function that sets specific row-values to NaN if 
        lower/higher than a value in specified columns in a pandas dataframe inplace

        Args:
            varaible: variable that the columns belong to. Here columns are single items.
            columns: list with column names in the dataframe 
            extremum: string, accepted values: 'min' or 'max'. Defining if finding values higher or lower than the extremum value.
            ext_val: in the case where there are several different allowed extremum values for a variable, can send in the extremum
                     value manually 
        
        Returns:
            None, but changes the dataframe inplace and adds the total number of changed values to total_count
        """

        count = 0
        
        # If extremum=min, then op is <, if extremum=max, op is >
        op = extremum

        if ext_val is None:
            extremum_value = self.cleanup_dict[variable][op]
        else:
            extremum_value = ext_val


        if isinstance(columns, pd.core.indexes.base.Index) or isinstance(columns, list):

            # idx = np.where(self.operators[op](self.df.loc[:,columns].values, extremum_value) == True)

            # self.df.loc[idx[0], idx[1]] = np.nan

            # count += len(idx[0])
            
            for col in columns:

                rows = np.where(self.operators[op](self.df.loc[:,col].values, extremum_value) == True)[0]
                
                count += len(rows)
                for row in rows:
                    self.df.loc[row,col] = np.nan
                    

        elif isinstance(columns, str):
            col = columns

            rows = np.where(self.operators[op](self.df.loc[:,col].values, extremum_value) == True)[0]
                
            count += len(rows)
            for row in rows:
                self.df.loc[row,col] = np.nan 
                

        else:
            print(f"Type {type(columns)} not recognized.")


    
        self.total_count += count

        if self.verbose == 1:

            b = "lower" if op == "min" else "higher"

            print(f"Identified {count} values {b} than {extremum_value} for {variable}-items. All set to NaN", end="\n")
            


    def _clean_LTHofMD(self, var):
        """
        Some special cleanup is required for the LTHofMD items. High number of NaN values in the third item. 
        This is a binary variable, so here we interpret the NaN values as "No"
        """           


        for Q in self.Q_names:
            try:
                variable_items = variables[Q][var]

                if Q == "Q1":
                    # pdb.set_trace()
                    binary_variables = variable_items[:-2] + [variable_items[-1]]
                    
                    self._set_values_nan(var, binary_variables, extremum="min", ext_val=1)           
                    self._set_values_nan(var, binary_variables, extremum="max", ext_val=2)


                    # Transforming binary encoding (1/2) --> (0/1)
                    self.df[binary_variables] = self.df[binary_variables].transform(np.vectorize(lambda x: x-1))

                    self.df.loc[:,variable_items[-1]] = self.df.loc[:,variable_items[-1]].fillna(0)
                    self.df["AA1578"] = self.df["AA1578"].fillna(0)

                    self.cleanup_dict[var]["min"] = 0

                elif Q == "Q6":
                    self._set_values_nan(var, variable_items[:6], extremum="min", ext_val=1)           
                    self._set_values_nan(var, variable_items[:6], extremum="max", ext_val=2)

                    self.df[variable_items[:6]] = self.df[variable_items[:6]].transform(np.vectorize(lambda x:x-1))

                    self.df[variable_items[6:12]].combine_first(self.df[variable_items[:6]])
                    self.df.drop(columns=self.df[variable_items[:6]], inplace=True)

                    del variable_items[:6]

                    self.df[variable_items] = self.df[variable_items].fillna(0)

        
            except KeyError:
                continue

    def _clean_rss(self, var):
        
        variable_items = self.items[var]

        self.df[variable_items] = self.df[variable_items].fillna(-1)

    def _clean_edu(self, var):
        """
        Changing NaN values to zero as they are most likely "no"s in the second education item.
        """

        second_item = self.items[var][1]

        # Changing NaN values to 0
        self.df.loc[:,second_item] = self.df.loc[:,second_item].fillna(0)

        self.cleanup_dict[var]["min"] = 0

    def _clean_imm(self, var):
        """
        Transforming binary encoding (1/2) --> (0/1)
        """
        variable_item = self.items[var]

        self.df[variable_item] = self.df[variable_item].transform(np.vectorize(lambda x: x-1))

        self.cleanup_dict[var]["min"] = 0

    def _clean_abuse(self, var):
        """
        Here there are a number of NaN values that should have been encoded as "no".
        So the function fixes this by replacing the NaN -> 1.
        """

        variable_items = self.items[var]

        # List containing the three items for each question
        Qs = [variable_items[i:i+3] for i in range(0,12,3)]

        # All indicies in the dataframe
        idx = np.arange(self.df.shape[0])

        # Nested list over participants that have NaN values in all of the items per question
        all_nans = [np.where(self.df[Q].sum(axis=1) == 0)[0] for Q in Qs]

        for i, Q in enumerate(Qs):
            ind = np.copy(idx)
            
            # Removing all participants that have NaN in all three items. The remaining NaN values are changed to 0
            ind = np.delete(ind, all_nans[i]) 
            
            for j in Q:
                self.df[j].iloc[ind] = self.df[j].iloc[ind].fillna(0)

    def _clean_ALE(self, var):
        """
        All of the items that asks for other events have a higher percentage of NaN-values. 
        Since it is a binary variable we interpret the NaN-values as "no". So the function changes
        NaN to 'no' for the selected columns
        """
        variable_items = self.items[var]

        other_col = ["CC1249", "DD825", "EE669", "GG540"]

        self.df.loc[:,other_col] = self.df.loc[:,other_col].fillna(1)

        # Transforming binary encoding (1/2) --> (0/1)

        self.df[variable_items] = self.df[variable_items].transform(np.vectorize(lambda x: x-1))

        self.cleanup_dict[var]["min"] = 0

    def _clean_birth(self, var):
        """
        Second item is a binary variable. Transforming the binary encoding from (1/2) to (0/1)
        """
        second_item = self.items[var][1]

        self.df[second_item] = self.df[second_item].transform(np.vectorize(lambda x: x-1))

        self.cleanup_dict[var]["min"][1] = 0

    def _clean_workQ3(self, var):

        variable_items = self.items[var]
        self.df["CC910"].loc[np.where(self.df["CC910"] == 0)[0]] = np.nan
        self.df["CC910"] = self.df["CC910"].transform(np.vectorize(lambda x: x-1))
        self.df["CC914"] = self.df["CC914"].transform(np.vectorize(lambda x: x-1))
        self.df[["CC911", "CC912", "CC913", "CC915"]] = self.df[["CC911", "CC912", "CC913", "CC915"]].fillna(-1)
        self.df[variable_items[7:]] = self.df[variable_items[7:]].fillna(0)
    
    def _clean_workQ6(self, var):

        variable_items = self.items[var]

        self._set_values_nan(var, variable_items[0], extremum="min", ext_val=1)           
        self._set_values_nan(var, variable_items[0], extremum="max", ext_val=2)

        self.df[variable_items[0]] = self.df[variable_items[0]].transform(np.vectorize(lambda x: x-1))

        self.df[variable_items[2:7]] = self.df[variable_items[2:7]].fillna(-1)
        self.df[variable_items[7]] = self.df[variable_items[7]].fillna(0)
        self.df[variable_items[8:]] = self.df[variable_items[8:]].fillna(-1)

    def _clean_sickleaveQ3(self, var):
        variable_items = self.items[var]
        self.df["CC937"] = self.df["CC937"].fillna(-1)
        self.df[variable_items[3:]] = self.df[variable_items[3:]].fillna(0)


        for col in ["CC946", "CC954", "CC962", "CC970"]:
            idx = np.where(self.df[col] > 100)[0]

            self.df.loc[idx, col] = 100

    def _clean_sickleaveQ4(self, var):
        variable_items = self.items[var]

        self.df[variable_items[1:]] = self.df[variable_items[1:]].fillna(0)


        for col in ["DD680", "DD686", "DD692"]:
            idx = np.where(self.df[col] > 100)[0]

            self.df.loc[idx, col] = 100

    def _clean_drugs(self, var):
        
        for Q in self.Q_names:
            try:
                variable_items = self.variables[Q][var]
                if Q == "Q3":
                    self.df[variable_items[:5]] = self.df[variable_items[:5]].transform(np.vectorize(lambda x: x-1))
                elif Q == "Q4":
                    self.df[variable_items] = self.df[variable_items].fillna(0)
                    # for i in range(0, len(variable_items)-1, 3):
                    #     self.df[variable_items[i+1]] = self.df[variable_items[i+1]].fillna(0)
                    #     self.df[variable_items[i+2]] = self.df[variable_items[i+2]].fillna(0)
        
            except KeyError:
                continue
    
    def _clean_alcohol(self, var):

        for Q in self.Q_names:
            try:
                variable_items = self.variables[Q][var]
                if Q == "Q3":

                    self._set_values_nan(var, variable_items[:4], extremum="min", ext_val=1)           
                    self._set_values_nan(var, variable_items[:4], extremum="max", ext_val=7)
                   
                    self._set_values_nan(var, variable_items[4:8], extremum="min", ext_val=1)           
                    self._set_values_nan(var, variable_items[4:8], extremum="max", ext_val=5)
                    
                    self._set_values_nan(var, variable_items[8:12], extremum="min", ext_val=1)
                    self._set_values_nan(var, variable_items[8:12], extremum="max", ext_val=6)

                    self.df[variable_items[8:12]] = self.df[variable_items[8:12]].fillna(7)
                    
                    self._set_values_nan(var, variable_items[12:17], extremum="min", ext_val=1)
                    self._set_values_nan(var, variable_items[12:17], extremum="max", ext_val=2)

                    self.df[variable_items[12:17]] = self.df[variable_items[12:17]].transform(np.vectorize(lambda x:x-1))
                    self.df[variable_items[12:17]] = self.df[variable_items[12:17]].fillna(0)

                elif Q == "Q4":
                    self._set_values_nan(var, variable_items[:3], extremum="min", ext_val=1)           
                    self._set_values_nan(var, variable_items[:3], extremum="max", ext_val=7)

                    self._set_values_nan(var, variable_items[3:], extremum="min", ext_val=1)           
                    self._set_values_nan(var, variable_items[3:], extremum="max", ext_val=6)

                    self.df[variable_items[3:]] = self.df[variable_items[3:]].fillna(7)

        
            except KeyError:
                continue

        
    def _clean_adverselifeevents(self, var):
         
        for Q in self.Q_names:
            try:
                variable_items = self.variables[Q]["AdverseLifeEvents"]
                
                for i in range(0,len(variable_items)-1,2):

                    self.df[variable_items[i]] = self.df[variable_items[i]].transform(np.vectorize(lambda x: x-1))
                    self.df[variable_items[i+1]] = self.df[variable_items[i+1]].fillna(-1)

            except KeyError:
                continue
   
    def _clean_assaultQ3(self, var):

        variable_items = self.items[var]
      
        for i in range(0, len(variable_items)-1, 7):
            item = variable_items[i:i+7]
            self.df[item[:-1]] = self.df[item[:-1]].fillna(0)
            self.df[item[-1]] = self.df[item[-1]].transform(np.vectorize(lambda x: x-1))
            self.df[item[-1]] = self.df[item[-1]].fillna(-1)

    def _clean_assaultQ6(self, var):
        
        variable_items = self.items[var]
        self.df[variable_items] = self.df[variable_items].transform(np.vectorize(lambda x: x-1))

    def _clean_Birth(self, var):

        self.df[["DD20", "DD21", "DD28"]] = self.df[["DD20", "DD21", "DD28"]].transform(np.vectorize(lambda x: x-1))
        self.df[["DD17", "DD21", "DD22", "DD23", "DD24", "DD25", "DD26", "DD27"]] = self.df[["DD17", "DD21", "DD22", "DD23", "DD24", "DD25", "DD26", "DD27"]].fillna(-1)
        self.df[["DD39", "DD40", "DD41"]] = self.df[["DD39", "DD40", "DD41"]].fillna(0)
    
    def _clean_childcare(self, var):

        variable_items = self.items[var]

        self.df[variable_items[:5]] = self.df[variable_items[:5]].fillna(0)

        self.df[variable_items[6]] = self.df[variable_items[6]].transform(np.vectorize(lambda x: x-1))
        self.df[variable_items[7]] = self.df[variable_items[7]].fillna(-1)

    def _clean_livingsituation(self, var):


        col = "EE916"
        rows = np.where(self.df.loc[:,col].values < 1)[0]
        
        for row in rows:
            self.df.loc[row,col] = np.nan


        self.df[col] = self.df[col].transform(np.vectorize(lambda x:x-1))

        col1 = [f"EE{i}" for i in range(492,496)]
        self.df[col1] = self.df[col1].fillna(0)

        col2 =  [f"EE{i}" for i in range(917,924)]
        self.df[col2] = self.df[col2].fillna(-1)

    def _clean_livingenvironment(self, var):

        self.df.drop(columns=["EE500", "EE947", "EE948", "EE949", "EE501", "EE502", "EE503", "EE504", "EE505"], inplace=True)


        variable_items = self.items[var]

        self.df[variable_items[:3]] = self.df[variable_items[:3]].fillna(0)
        self.df[variable_items[-1]] = self.df[variable_items[-1]].fillna(-1)

    def _clean_parentalleave(self, var):

        self._set_values_nan(var, "EE577", extremum="max", ext_val=24*7)         

        self.df[["EE572", "EE573", "EE574", "EE575", "EE577"]] = self.df[["EE572", "EE573", "EE574", "EE575", "EE577"]].fillna(-1)

        self.df["EE576"] = self.df["EE576"].transform(np.vectorize(lambda x: x-1))

        self.df[["EE578", "EE579", "EE580", "EE581", "EE582"]] = self.df[["EE578", "EE579", "EE580", "EE581", "EE582"]].fillna(-1)

    def _clean_smokingQ4(self, var):
        
        variable_items = self.items[var]

        for item in variable_items[:-2]:
            self._set_values_nan(var, item, extremum="max", ext_val=1)

        self.df[variable_items[:-2]] = self.df[variable_items[:-2]].fillna(0)
        self.df[variable_items[-1]] = self.df[variable_items[-1]].fillna(0)

        self._set_values_nan(var, variable_items[-2], extremum="min", ext_val=1)           
        self._set_values_nan(var, variable_items[-2], extremum="max", ext_val=4)

    def _clean_smokingQ5(self, var):
        self.df[["EE604", "EE606"]] = self.df[["EE604", "EE606"]].fillna(-1)
        self.df["EE609"] = self.df["EE609"].fillna(7)
   
    def _clean_smokingQ6(self, var):
        
        variable_items = self.items[var]

        self._set_values_nan(var, variable_items[0], extremum="min", ext_val=1)           
        self._set_values_nan(var, variable_items[0], extremum="max", ext_val=3)

        self.df[variable_items[1:]] = self.df[variable_items[1:]].fillna(0)
   
    def _clean_childlengthweight(self, var):

        for Q in self.Q_names:
            try:
                variable_items = self.variables[Q][var]

                if Q == "Q5":
                    pass
                    # remove_cols = ["Q5_AGE_8_M", "Q5_AGE_1_Y", "Q5_AGE_15_18_M"]
                    # self.df.drop(columns=remove_cols, inplace=True)


                elif Q == "Q6":
                    
                    remove_cols = ["Q6_AGE_3_Y", "Q6_AGE_2_Y", "Q6_AGE_18_M", "GG664", "GG665", "GG666"]
                    self.df.drop(columns=remove_cols, inplace=True)

            except KeyError:
                continue

    def _clean_commnication(self, var):
        for Q in self.Q_names:
            try:
                variable_items = self.variables[Q][var]

                if Q == "Q5":
                    for col in variable_items:

                        rows = np.where(self.df.loc[:,col].values < 1)[0]
                    
                        for row in rows:
                            self.df.loc[row,col] = np.nan

                    for col in variable_items:

                        rows = np.where(self.df.loc[:,col].values > 3)[0]
                    
                        for row in rows:
                            self.df.loc[row,col] = np.nan
                
                elif Q == "Q6":
                    rows = np.where(self.df.loc[:,variable_items[0]].values < 1)[0]
                    print(len(rows))
                    for row in rows:
                        self.df.loc[row,variable_items[0]] = np.nan
                    
                    rows = np.where(self.df.loc[:,variable_items[0]].values > 6)[0]
                    print(len(rows))
                    for row in rows:
                        self.df.loc[row,variable_items[0]] = np.nan
                        
                    for col in variable_items[1:]:
                        
                        rows = np.where(self.df.loc[:,col].values > 3)[0]
                        print(len(rows))
                        for row in rows:
                            self.df.loc[row,col] = np.nan

                        rows = np.where(self.df.loc[:,col].values <1)[0]
                        print(len(rows))
                        for row in rows:
                            self.df.loc[row,col] = np.nan



            except KeyError:
                continue

    def _clean_autism(self, var):
         for Q in self.Q_names:
            try:

                variable_items = self.variables[Q][var]
                
                if Q == "Q5":
                
                    for var in variable_items:
                        try:
                            self.df.drop(columns=[var], inplace=True)
                        except KeyError:
                            continue

                elif Q == "Q6":
                    col = "GG252"
                    
                    try:
                        self.df.drop(columns=[col], inplace=True)
                    except KeyError:
                        pass

                    variable_items.remove(col)

                    self._set_values_nan(var, variable_items, extremum="min", ext_val=1)           
                    self._set_values_nan(var, variable_items, extremum="max", ext_val=2)

                    self.df[variable_items] = self.df[variable_items].transform(np.vectorize(lambda x:x-1))
               


            except KeyError:
                continue

    def _clean_daycare(self, var):

         for Q in self.Q_names:
            try:
                variable_items = self.variables[Q][var]

                if Q == "Q5":

                    self.df[variable_items] = self.df[variable_items].fillna(-1)


                elif Q == "Q6":
                    
                    self.df[variable_items[:-1]] = self.df[variable_items[:-1]].fillna(-1) 


            except KeyError:
                continue

    def _clean_livingwithfather(self, var):
        
        
        for Q in self.Q_names:
            try:
                variable_items = self.variables[Q][var]

                if Q == "Q5":

                    min_val = [1,1]
                    max_val = [2,5]
                    for i, col in enumerate(variable_items):
                        self._set_values_nan(var, col, extremum="min", ext_val=min_val[i])           
                        self._set_values_nan(var, col, extremum="max", ext_val=max_val[i])           
               
                    self.df[variable_items[0]] = self.df[variable_items[0]].transform(np.vectorize(lambda x:x-1))
                    self.df[variable_items[-1]] = self.df[variable_items[-1]].fillna(-1)


                elif Q == "Q6":
                    
                    min_val = [1,1,1]
                    max_val = [2,6,6]
                    for i, col in enumerate(variable_items):
                        self._set_values_nan(var, col, extremum="min", ext_val=min_val[i])           
                        self._set_values_nan(var, col, extremum="max", ext_val=max_val[i])           
               
                    self.df[variable_items[0]] = self.df[variable_items[0]].transform(np.vectorize(lambda x:x-1))
                    self.df[variable_items[1:]] = self.df[variable_items[1:]].fillna(-1)

            except KeyError:
                continue

    def _clean_timeoutside(self, var):
        
        
        for Q in self.Q_names:
            try:
                variable_items = self.variables[Q][var]

                if Q == "Q5":

                    min_val = [1,1,1,0]
                    max_val = [4,5,2,999]
                    for i, col in enumerate(variable_items):
                        self._set_values_nan(var, col, extremum="min", ext_val=min_val[i])           
                        self._set_values_nan(var, col, extremum="max", ext_val=max_val[i])           
               
                    self.df[variable_items[2]] = self.df[variable_items[2]].transform(np.vectorize(lambda x:x-1))
                    self.df[variable_items[3]] = self.df[variable_items[3]].fillna(-1)


                elif Q == "Q6":
                    
                    min_val = [1,1]
                    max_val = [4,5]
                    for i, col in enumerate(variable_items):
                        self._set_values_nan(var, col, extremum="min", ext_val=min_val[i])           
                        self._set_values_nan(var, col, extremum="max", ext_val=max_val[i])           
         



            except KeyError:
                continue

    def _clean_pregnantnow(self, var):

        variable_items = self.items[var]

        self._set_values_nan(var, variable_items[0], extremum="min", ext_val=1)           
        self._set_values_nan(var, variable_items[0], extremum="max", ext_val=2)

        self.df[variable_items[0]] = self.df[variable_items[0]].transform(np.vectorize(lambda x:x-1))
        self.df[variable_items[-1]] = self.df[variable_items[-1]].fillna(-1)

    def _clean_walkunaided(self, var):

        variable_items = self.items[var]

        self.df[variable_items[-1]] = self.df[variable_items[-1]].fillna(-1)

    def _clean_socialcommunication(self, var):

        variable_items = self.items[var]

        self.df[variable_items] = self.df[variable_items].transform(np.vectorize(lambda x:x-1))

    def _clean_lifting(self, var):

        variable_items = self.items[var]

        df[variable_items[1]] = df[variable_items[1]].transform(np.vectorize(lambda x:x-1))
        df[variable_items[1]] = df[variable_items[1]].fillna(0)


    def summary(self):
        """
        Prints some summary statistics
        """

        print_info("Printing summary statistics \n")

        tot_nan = sum(self.df.isna().sum())

        tot_responses = self.df.shape[0]*self.df.shape[1]

        percent_nan = tot_nan/tot_responses * 100

        rows_with_nan = self.df.isna().any(axis=1).sum()

        tot_rows_nan = rows_with_nan / self.df.shape[0] * 100

        percent_nan_col = (self.df.isna().sum(axis=0)/self.df.shape[0] * 100) 

        data = [
            ["Total values changed to NaN:", self.total_count],
            ["Total NaN-values in the whole dataframe", tot_nan],
            ["Total item responses:", tot_responses],
            ["Percentage of NaN values in the whole dataframe:", percent_nan],
            ["Percentage of rows with NaN values:", tot_rows_nan]
        ]

        print("-"*80)
        for i in range(len(data)):
            if i == 3 or i == 4:
                print("{:.<55s}{:.2f}%\n".format(data[i][0], data[i][1]))
            else:
                print("{:.<55s}{:.2f}\n".format(data[i][0], data[i][1]))


        
        print("Columns with more than 10% of values being NaN:")

        # Check if any columns has more than 10% of NaN-values
        label = False
        for i, col in enumerate(percent_nan_col):
            if col > 10:
                print(df.columns[i])
                label = True
        if not label:
            print("No columns with more than 10% of NaN-values")
        print("-"*80, end="\n")
    
        

if __name__ == '__main__':


    # with open("./../data/variables.hjson", "r") as f:
    #     variables = hjson.load(f)
    
    with open("./../data/all_variables.json", "r") as f:
        variables = json.load(f)

    # variable_keys = [
    #     "scl",
    #     "LTHofMD",
    #     "RSS",
    #     "ss",
    #     "edu",
    #     "imm",
    #     "abuse",
    #     "income",
    #     "GSE",
    #     "anger",
    #     "RSES",
    #     "SWLS",
    #     "ALE",
    #     "birth"
    # ]

    variable_keys = ["PregnantNow"]
        
        
    Q_names = ['Q1', 'Q3', 'Q4', 'Q5', 'Q6']



    data_path = "./../experiments/{}/dataframes/".format("_".join(Q_names))
    var_qs_name = (
                "_".join(variable_keys)
                + "_"
                + "_".join([Q for Q in variables.keys() if Q in Q_names])
                + ".csv"
            )

    df_name = "sub_dataframe_" + var_qs_name

    df_name = "cleaned_dataframe_" + "_".join(Q_names) + ".csv"



    df = pd.read_csv(data_path + df_name, delimiter=";")
    df.to_csv(data_path + df_name, index=False)
    # pdb.set_trace()
    # cleaner = Cleaner(df, variable_keys, variables, Q_names)
    # cleaner()




    # for Q in Q_names:
    #     try:
    #         for i in range(len(variable_keys)):
    #             items = variables[Q][variable_keys[i]]
    #             print(df[items].describe())
    #             print(df[items].isna().sum() / (df.shape[0]) *100)
    #     except KeyError:
    #         continue
    
    # pdb.set_trace()
    # cleaner.df.to_csv(data_path + df_name, index=False)