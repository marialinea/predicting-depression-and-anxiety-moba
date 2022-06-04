import numpy as np
import pandas as pd
import json
import datetime
import itertools
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import re
import pdb

from utils.configure import Configure, get_args
from utils.data import prepare_folds
from utils.print import print_info


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

from scipy.stats import uniform
from sklearn.utils.fixes import loguniform



class ElasticoNet(Configure):

    def __init__(self, args, algorithm="elastic_net", subsample=False):

        super().__init__(args, algorithm)

        self.subsample = subsample
        self._en_config()

    def _en_config(self):
        """
        Helper function to configure the specific elastic net class
        """

        # Path to config files
        self.config_path = os.path.join(self.root, self.algorithm, "configs")

        self._check_dir(self.config_path)
  
        if self.args.config is None:

            # Name of config file
            self.config_name = f"config_{self.unique_id}.json"



    def tune(self):
        """
        Tune tree structure with RandomizedSearchCV from scikit-learn. 
        The results are stored in a json-file and saved in the config directory.
        """

        print("")
        print_info("Tuning Elastic Net \n")

        train_df = self.train_test_dict["train"]
        test_df = self.train_test_dict["test"]

      
        # Split data
        X_train, X_test, y_train, y_test = prepare_folds(train_df, test_df, 
                                                            self.target_column, 
                                                            remove_cols=self.remove_cols, 
                                                            validation=False, 
                                                            remove_nan=False)

        # Scaling design matrices
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        if self.subsample is True:
            X_train, y_train = self.subsample_procedure(X_train, y_train, ratio=0.1)

        
        # Defining Model
        model = ElasticNet()
    
        
        n_iter = 30

        
        if self.PCA is True:

            pca = PCA(n_components=0.95, random_state=42)

            X_train = pca.fit_transform(X_train)

    
        param_dist = {
                "alpha": uniform(1e-4, 1),
                "l1_ratio": uniform(0,1)
        }

        scoring = "neg_root_mean_squared_error"

        # Perform randomized search 
        grid = RandomizedSearchCV(
                                    model, 
                                    param_distributions=param_dist, 
                                    scoring=scoring,
                                    n_iter=n_iter,  
                                    verbose=2,
                                    random_state=int(datetime.datetime.now().strftime("%H%M%S"))
        )
        
    
        grid_result = grid.fit(X_train, y_train)

        config_dict = grid.best_params_
        
        if self.experiment in self.long_experiments:
            config_dict["Q"] = self.Q_names[-1]

        if self.PCA is True:

            num_components = pca.n_components_
            config_dict["num_componenets"] = int(num_components)

        with open(f"{self.config_path}/{self.config_name}", "w") as outfile:
            json.dump(config_dict, outfile, indent=4)
    

    def subsample_procedure(self, X_train, y_train, ratio):

        majority_group_train = np.where(y_train < 1.75)[0]

        minority_group_train = np.where(y_train > 1.75)[0]
   
        np.random.seed(10)
        idx_train = np.random.choice(majority_group_train, size=int(ratio*len(majority_group_train)), replace=False)
        
        include_train = np.concatenate([minority_group_train, idx_train])
        
        X_train = X_train[include_train, :]
        y_train = y_train[include_train]

        return X_train, y_train

    def train(self, save_preds=False):

        # Load best parameters
        with open(f"{self.config_path}/{self.config_name}", "r") as infile:
            self.config = json.load(infile)

        print("")
        print_info("Start Training \n")



        # Dictionary to store the results w.r.t to metrics
        results = {}
        
        train_df = self.train_test_dict["train"]
        test_df = self.train_test_dict["test"]


        if self.PCA is True:
            mse, rmse, mae, residuals, num_components = self.train_eval_one(train_df, test_df, save_preds=save_preds)
        
            results["num_components"] = num_components
        else:
            mse, rmse, mae, residuals = self.train_eval_one(train_df, test_df, save_preds=save_preds)


        results["rmse"] = rmse
        results["mse"] = mse
        results["mae"] = mae

            
        # np.save(f"{self.result_path}/en_residuals_{self.unique_id}.npy", residuals)
        with open(f"{self.result_path}/en_metrics_{self.unique_id}.json", "w") as outfile:
            json.dump(results, outfile, indent=4)



    def train_eval_one(self, train_df, test_df, save_preds, Q=None):
       
        result_name = f"preds_true_{self.unique_id}.npy"

        # Split data 
        X_train, X_test, y_train, y_test = prepare_folds(
                                                        train_df, test_df, 
                                                        self.target_column, 
                                                        remove_cols=self.remove_cols,
                                                        validation=False, 
                                                        remove_nan=False)
    
        # Standardize input data
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        if self.subsample is True:
            X_train, y_train = self.subsample_procedure(X_train, y_train, ratio=0.3)
            # X_test, y_test = self.subsample_procedure(X_test, y_test, ratio=0.01)
        
        # pdb.set_trace()
        # List over independent variables
        independent_vars = test_df.columns

        
        try:
            independent_vars = independent_vars.drop(self.remove_cols)
        except KeyError:
            pass
       
        # Defining model parameters
        alpha = self.config["alpha"]
        l1_ratio = self.config["l1_ratio"]


        
        # Setting up model
        model = ElasticNet(alpha=alpha,
                            l1_ratio=l1_ratio
        )            

        if self.PCA is True:

            pca = PCA(n_components=0.95, random_state=42)

            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            num_components = int(pca.n_components_)

    
        # Fit model
        history = model.fit(X_train, y_train)

        
        # Evaluate and make predicitons
        preds = model.predict(X_test)

        # Saving Results
        preds_true = np.column_stack((preds, y_test))
        
        if save_preds is True:
            np.save(os.path.join(self.result_path, result_name), preds_true)
        
        mse = mean_squared_error(preds, y_test)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - preds))
        residuals = y_test - preds

    
        if self.PCA is True:

            principal_components = pca.components_

            pca_name = f"en_principle_components_{self.unique_id}.npy"
            np.save(os.path.join(self.coeff_path, pca_name), principal_components)
        

        # Saving coefficients
        coeffs = {}
        if self.PCA is False:
            variables = independent_vars
            index_name = "features"
        else:
            variables = [f"{i+1}" for i in range(principal_components.shape[0])]
            index_name = "principal_comps"
        

        for var, coef in zip(variables, model.coef_):
            coeffs[var] = coef
      
        summary_df = pd.DataFrame.from_dict(coeffs, orient="index")
        summary_df.columns = ["coef"]
        summary_df.index.name = index_name

        coeff_path = f"{self.coeff_path}/en_coefficients_{self.unique_id}.csv"
        summary_df.to_csv(coeff_path)
        
        

        if self.PCA is True:
            return mse, rmse, mae, residuals, num_components
        else:
            return mse, rmse, mae, residuals


def main():
    
    # Get commandline arguments
    args = get_args()
    
    subsample = False

    if args.mode == "tune":


        # If passed a config name when mode=tune, this is overwritten
        if args.config is not None:
            args.config = None
        # Create class instance
        en = ElasticoNet(args, subsample=subsample)
        # pdb.set_trace()

        # Call tuning function
        en.tune()

        en.train(save_preds=True)

    elif args.mode == "train" or args.mode == "train_eval":

        # Create class instance
        en = ElasticoNet(args, subsample=subsample)

        # Call train function
        en.train(save_preds=True)
    else:
        raise ValueError

if __name__=='__main__':

    
    main()
