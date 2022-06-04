import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import joblib
import datetime

import os
import re
import sys
import pdb

from utils.configure import Configure, get_args
from utils.data import prepare_folds
from utils.print import print_info

from tensorboardX import SummaryWriter
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils.fixes import loguniform
from sklearn.decomposition import PCA



class Boosting(Configure):

    def __init__(self, args, algorithm="boosting"):

        super().__init__(args, algorithm)

        self._xgb_config()
       

    def _xgb_config(self):
        """
        Helper function to configure the specific boosting class
        """

        # Path to logs
        self.log_path = os.path.join(self.root, self.algorithm, "logs")

        self._check_dir(self.log_path)

        # Path to config files
        self.config_path = os.path.join(self.root, self.algorithm, "configs")

        self._check_dir(self.config_path)



     


    def tune(self):
        """
        Tune tree structure with RandomizedSearchCV from scikit-learn. 
        The results are stored in a json-file and saved in the config directory.
        """

        print("")
        print_info("Tuning Tree \n")


        train_df = self.train_test_dict["train"]
        test_df = self.train_test_dict["test"]
    
        # Split data
        X_train, _, y_train, _ = prepare_folds(
                                                train_df,
                                                test_df,
                                                (self.target_column),
                                                remove_cols=(self.remove_cols),
                                                validation=False,
                                                remove_nan=False,
        )
 
        
        X_train = StandardScaler().fit_transform(X_train)
 
        if self.PCA is True:

            pca = PCA(n_components=0.95, random_state=42)

            X_train = pca.fit_transform(X_train)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


        fit_params = {
            "early_stopping_rounds": 1000,
            "eval_metric": "rmse",
            "eval_set": [(X_val, y_val)]
        }


        # Setting parameters to tune
        param_grid = {
            # Parameters we are going to tune
            "max_depth": [i for i in range(2,8)],
            "min_child_weight": [i for i in range(1,6)],
            "eta": [0.01, 0.005],
            "subsample": [i/10. for i in range(1,6)],
            "colsample_bytree": [i/10. for i in range(1,6)],
            "n_estimators": [1000, 5000, 10_000, 20_000],
            "objective": ["reg:squarederror"]
        }     
        
        n_iter = 15
            

        # Defining Model
        model = xgb.XGBRegressor(random_state=42)

        scoring = "neg_root_mean_squared_error"

        grid = RandomizedSearchCV(estimator=model,
                                    param_distributions=param_grid, 
                                    scoring=scoring,
                                    cv=5,
                                    n_iter=n_iter,
                                    verbose=1,
                                    random_state=int(datetime.datetime.now().strftime("%H%M%S"))
                                    )
            
        # Perform randomized search 
        grid_result = grid.fit(X_train,y_train, **fit_params)

        config_dict = grid.best_params_
        
        config_dict["scoring"] = scoring

        if self.PCA is True:
            config_dict["num_components"] = int(pca.n_components_)

        with open(f"{self.config_path}/{self.config_name}", "w") as outfile:
            json.dump(config_dict, outfile, indent=4)

        print(grid.best_params_)
        print(grid.cv_results_)


    def train(self, save_preds=False):

        # Load best parameters
        with open(f"{self.config_path}/{self.config_name}", "r") as infile:
            self.config = json.load(infile)


        print("")
        print_info("Start Training \n")


        # Dictionary to store the results w.r.t to metrics
        results = {}


        result_name = f"preds_true_{self.unique_id}.npy"

        train_df = self.train_test_dict["train"]
        test_df = self.train_test_dict["test"]

        self.train_one(train_df, test_df)



    def train_one(self, train_df, test_df, Q=None):

        
        self._prepare_data(train_df, test_df)
            

        num_boost_round = self.config["n_estimators"]
        early_stopping = int(num_boost_round/4)
        
        # Defining model parameters
        params = {
            "max_depth"          : self.config["max_depth"],
            "min_child_weight"   : self.config["min_child_weight"],
            "eta"                : self.config["eta"],
            "subsample"          : self.config["subsample"],
            "colsample_bytree"   : self.config["colsample_bytree"],
            "objective"          : self.config["objective"]
        }

        log_dir = os.path.join(self.log_path, f"logs_{self.unique_id}")
           
        # Setting up model
        self.model = xgb.train(
            params,
            self.dtrain,
            num_boost_round= num_boost_round,
            evals = [(self.dval, "Eval")],
            callbacks=[TensorboardCallback(log_dir)],
            early_stopping_rounds=early_stopping
        )            


        self._extract_feature_importances(train_df,Q)


        # Saving model

        model_name = f"model_{self.unique_id}.joblib.dat"

        joblib.dump(self.model, os.path.join(self.model_path,model_name))



    def make_plot(self, num_features=20):

        print_info("Constructing feature importance plot")

        fig_path = f"{self.root}/{self.algorithm}/figures"
        
        self._check_dir(fig_path)

        if self.experiment in self.long_experiments:

            for Q in self.train_test_dict.keys():

                fig_name = f"feature_importance_gain_{self.target}_scl_squarederror_{Q}.pdf"

                feature_name = f"feature_importance_{Q}.csv"

                features_and_importance = pd.read_csv(os.path.join(self.unique_feature_path, feature_name))

                # pdb.set_trace()

                if "Unnamed: 0" in features_and_importance.columns:
                    features_and_importance = features_and_importance.rename(columns={"Unnamed: 0": "features"})
                    features_and_importance = features_and_importance.set_index("features")
                    features_and_importance.to_csv(os.path.join(self.unique_feature_path, feature_name), index=True)

                # features = list(features_and_importance.index)
                # importance = features_and_importance.iloc[:,0].to_numpy()

                figure_name_path = os.path.join(fig_path, fig_name)

                self._plot(features_and_importance, figure_name_path, num_features)

        elif "wide" in self.experiment:

            fig_name = f"feature_importance_gain_{self.unique_id}.pdf"

            feature_name = f"feature_importance.csv"

            features_and_importance = pd.read_csv(os.path.join(self.unique_feature_path, feature_name))

            if "Unnamed: 0" in features_and_importance.columns:
                features_and_importance = features_and_importance.rename(columns={"Unnamed: 0": "features"})
                features_and_importance = features_and_importance.set_index("features")
                features_and_importance.to_csv(os.path.join(self.unique_feature_path, feature_name), index=True)

            # features = list(features_and_importance.index)
            # importance = features_and_importance.iloc[:,0].to_numpy() 

            figure_name_path = os.path.join(fig_path, fig_name)

            self._plot(features_and_importance, figure_name_path, num_features)

    def _plot(self, data_df , fig_path, num_features):

        FONTSIZE=14

        x = data_df.nlargest(num_features, columns="score")["features"]
        y = data_df.nlargest(num_features, columns="score")["score"]
        
        fig, ax = plt.subplots()

        sns.barplot(x=x, y=y, color="aquamarine")
        plt.ylabel("Gain", fontsize=FONTSIZE)
        plt.xlabel("Features", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE),
        plt.yticks(fontsize=FONTSIZE)
        plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right')

        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
       
    def _extract_feature_importances(self, train_df,Q):

        # Extract feature importance
        feature_importance_dict = self.model.get_score(importance_type="gain")
        importance = list(feature_importance_dict.values())
        features = list(feature_importance_dict.keys())

        
        # Match name from xgboost model with column names
        feature_index = [int(f"{feature[1:]}") for feature in features]
        if self.PCA is True:
            features = feature_index
        else:
            columns = list(train_df.columns)
    
            if self.remove_cols is not None:
                for i in self.remove_cols:
                    try:
                        columns.remove(i)
                    except ValueError:
                        continue

            columns = np.array(columns)
            features = columns[feature_index]
        
        

        data = pd.DataFrame(data=importance, index=features, columns=["score"]).sort_values(by="score", ascending=False)

        if Q is not None:
            feature_name = f"feature_importance_{Q}.csv"
        else:
            feature_name = "feature_importance.csv"
            
        data.to_csv(os.path.join(self.unique_feature_path, feature_name), index=True)


    def _prepare_data(self, train_df, test_df):

        # Split data 
        self.X_train, X_val, X_test, self.y_train, y_val, self.y_test = prepare_folds(
                                                            train_df, test_df, 
                                                            self.target_column, 
                                                            remove_cols=self.remove_cols,
                                                            validation=True, 
                                                            remove_nan=False)
        #################################
        # majority_group_train = np.where(y_train < 1.75)[0]
        # # majority_group_val = np.where(self.y_val < 1.75)[0]

        # minority_group_train = np.where(y_train > 1.75)[0]
        # # minority_group_val = np.where(self.y_val > 1.75)[0]
        
        # np.random.seed(0)
        # idx_train = np.random.choice(majority_group_train, size=int(0.5*len(majority_group_train)), replace=False)
        # # idx_val = np.random.choice(majority_group_val, size=int(0.5*len(majority_group_val)), replace=False)

        # include_train = np.concatenate([minority_group_train, idx_train])
        # # include_val = np.concat([minority_group_val, idx_val])
        
        # self.X_train = self.X_train.iloc[include_train, :]
        # y_train = y_train[include_train]
        # # X_val = X_val[include_val, :]

        #################################

        # Standardize input data
        self.X_train = StandardScaler().fit_transform(self.X_train)
        X_val = StandardScaler().fit_transform(X_val)
        X_test = StandardScaler().fit_transform(X_test)

        if self.PCA is True:
            pca = PCA(n_components=0.95, random_state=42)
            pca.fit(self.X_train)

            num_components = int(pca.n_components_)

            self.X_train = pca.transform(self.X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)

        # Converting data to xgboost objects
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dval = xgb.DMatrix(X_val, label=y_val)
        self.dtest = xgb.DMatrix(X_test, label=self.y_test)

        
    def eval(self, save_preds=False):

        print_info("Evaluating Model \n")

        results = {}

        train_df = self.train_test_dict["train"]
        test_df = self.train_test_dict["test"]
        
        mse, rmse, mae, residuals = self.evaluate_one(train_df, test_df, Q=None, save_preds=save_preds)

        
        results["rmse"] = rmse
        results["mse"] = mse
        results["mae"] = mae
        


        # np.save(f"{self.result_path}/xgb_residuals_{self.unique_id}.npy", residuals)
        with open(f"{self.result_path}/boosting_metrics_{self.unique_id}.json", "w") as outfile:
            json.dump(results, outfile, indent=4)

            

    def evaluate_one(self, train_df, test_df, Q=None,save_preds=False):

        model_name = f"model_{self.unique_id}.joblib.dat"
        result_name = f"preds_true_{self.unique_id}.npy"


        model = joblib.load(os.path.join(self.model_path, model_name))

        
        self._prepare_data(train_df, test_df)

        preds = model.predict(self.dtest)

        preds_true = np.column_stack((preds, self.y_test))
                
        if save_preds is True:
            np.save(os.path.join(self.result_path, result_name), preds_true)
            
        mse = mean_squared_error(preds, self.y_test)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(self.y_test - preds))
        residuals = self.y_test - preds

        return mse, rmse, mae, residuals

def TensorboardCallback(dir):
    writer = SummaryWriter(dir)
    def callback(env):
        for k, v in env.evaluation_result_list:
            writer.add_scalar(k, v, env.iteration)
    return callback
    


def main():
    # Get commandline arguments
    args = get_args()

    if args.mode == "tune":

        # If passed a config name when mode=tune, this is overwritten
        if args.config is not None:
            args.config = None

        # Create class instance
        boosting = Boosting(args)

        boosting.tune()

        boosting.train()

        boosting.eval(save_preds=True)


    elif args.mode == "train" or args.mode == "train_eval":

        # Create class instance
        boosting = Boosting(args)
        
        boosting.train()

        if args.mode == "train_eval":

            boosting.eval(save_preds=True)

    

    elif args.mode == "eval":

        if args.config is None:
            raise ValueError("Must specify config when evaluating model")

        boosting = Boosting(args)

        boosting.eval(save_preds=True)
        
    elif args.mode == "plot":

        # pdb.set_trace()
        boosting = Boosting(args)

     
        boosting.make_plot()

    

    else:
        raise ValueError

if __name__=='__main__':

    
    main()







