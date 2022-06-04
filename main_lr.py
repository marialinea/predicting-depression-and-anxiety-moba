import numpy as np
import pandas as pd
import json
import datetime
import itertools
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
import re
import pdb

from utils.configure import Configure, get_args
from utils.data import prepare_folds
from utils.print import print_info


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


sys.setrecursionlimit(1500)

class LinearRegressor(Configure):

    def __init__(self, args, algorithm="linear_regression"):

        super().__init__(args, algorithm)

    def train(self, save_preds=False):

        
        # Dictionary to store the results w.r.t tometrics
        results = {}

        train_df = self.train_test_dict["train"]
        test_df = self.train_test_dict["test"]
        
        
        if self.PCA is True:

            mse, rmse, mae, residuals, num_components = self.train_one(train_df, test_df, save_preds=save_preds)

            results["num_components"] = num_components

        else:
            mse, rmse, mae, residuals = self.train_one(train_df, test_df, save_preds=save_preds)

        results["rmse"] = rmse
        results["mse"] = mse
        results["mae"] = mae
    
        # np.save(f"{self.result_path}/lr_residuals_{self.unique_id}.npy", residuals)
        with open(f"{self.result_path}/lr_metrics_{self.unique_id}.json", "w") as outfile:
            json.dump(results, outfile, indent=4)



    def train_one(self, train_df, test_df, save_preds, Q=None):
        
  
        result_name = f"preds_true_{self.unique_id}.npy"
      


        X_train, X_test, y_train, y_test = prepare_folds(
                                                        train_df,
                                                        test_df,
                                                        (self.target_column),
                                                        remove_cols=(self.remove_cols),
                                                        validation=False,
                                                        remove_nan=False,
        )

        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        if self.PCA is True:

            pca = PCA(n_components=0.95, random_state=42)

            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            
            num_components = int(pca.n_components_)
            
        
        fitted_regressor = LinearRegression().fit(X_train, y_train)
        

        
        preds = fitted_regressor.predict(X_test)
        
        # Saving Results

        preds_true = np.column_stack((preds, y_test))

        if save_preds is True:
            np.save(os.path.join(self.result_path, result_name), preds_true)

        mse = mean_squared_error(preds, y_test)
        rmse = np.sqrt( np.mean((preds - y_test)**2) )
        mae = np.mean(np.abs(y_test - preds))
        residuals = y_test - preds


        if self.PCA is True:
            
            principal_components = np.array(pca.components_,dtype=object)

            pca_name = f"lr_principle_componenets_{self.unique_id}.npy"

            np.save(os.path.join(self.coeff_path, pca_name), principal_components)
            
            index_name = "principal_comps"
            index = np.array([i for i in range(components.shape[0])])
        
        else:
            
            index_name = "features"
            index = test_df.drop(columns=["PREG_ID_2601", self.target_column]).columns 

        
        coefs = fitted_regressor.coef_.squeeze()

        summary_df = pd.DataFrame(data=[index, coefs]).T
        summary_df.columns =  [f"{index_name}", "coef"]

            
        summary_name = f"lr_coefficients_{self.unique_id}.csv"
 
        summary_df.to_csv(os.path.join(self.coeff_path, summary_name), index=False)

        if self.PCA is True:

            return mse, rmse, mae, residuals, num_components

        else:
            return mse, rmse, mae, residuals



def main():
    # Get commandline arguments
    args = get_args()

    # pdb.set_trace()
    lr = LinearRegressor(args)

    lr.train(save_preds=False)

if __name__=='__main__':

    
    main()
