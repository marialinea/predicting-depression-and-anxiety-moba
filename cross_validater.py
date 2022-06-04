import pandas as pd
import numpy as np
import os
import json
import pdb
import xgboost as xgb
# from xgboost.sklearn import XGBRegressor

from utils.configure import Configure, get_args
from utils.data import prepare_folds, paired_ttest_5x2cv 

from main_nn import NeuralNet

from sklearn import model_selection
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor




class CrossValidater(object):

    def __init__(self, args, Q):

        self.args = args
        self.experiment = args.experiment
        self.Q = Q

        # Questionnaires
        self.Q_names = [q.upper() if q.islower() else q for q in args.qs]

        self.Q_list = "_".join(self.Q_names)

        self.path = f"./experiments/{self.Q_list}/{self.experiment}_analysis/"
                
        self.get_configs()
        
        self.get_models()

        self.metric = "neg_root_mean_squared_error"

        self._get_dataframes()

        self.X_train, self.X_test, self.y_train, self.y_test = prepare_folds(
                                                            self.train_df, self.test_df, 
                                                            self.NN_instance.target_column, 
                                                            remove_cols=self.NN_instance.remove_cols,
                                                            validation=False, 
                                                            remove_nan=False
                                                            )

        self.X_train = StandardScaler().fit_transform(self.X_train)

        if "pca" in self.experiment:
            pca = PCA(n_components = 0.95, random_state=42)

            self.X_train = pca.fit_transform(self.X_train)



    def __call__(self, n_splits=10):

        self.CV(n_splits)


   

        # print(self.results)     # outfile_path = self.path + f"{n_splits}cv_results_{self.Q}.json"
        # with open(outfile_path, "w") as f:
        #     json.dump(self.results, f, indent=4)

    def get_configs(self):

        algorithms = ["linear_regression", "elastic_net", "neural_net", "boosting"]

        config_path = self.path + "{algorithm}/configs"

        self.hyperparameters = {
                "linear_regression": None,
                "elastic_net": [
                    "alpha", 
                    "l1_ratio"
                ],
                "neural_net": [
                    "opt", 
                    "lr",
                    "layer_sizes",
                    "l2_reg",
                    "input_shape",
                    "epochs",
                    "dropout_rate",
                    "dropout",
                    "batch_size",
                    "activation"
                ],
                "boosting": [
                    "subsample",
                        "objective",
                        "n_estimators",
                        "min_child_weight",
                        "max_depth",
                        "eta",
                        "colsample_bytree"
                ]
            }

        self.configs = {}

        self.fit_params = {}
        
        


        if self.experiment == "wide_format":

            self.config_names = {
                    "linear_regression": None,
                    "elastic_net": os.path.join(config_path.format( 
                        algorithm="elastic_net"),
                        "config_19.04.2022-175857.json"),
                    "neural_net": os.path.join(config_path.format(
                        algorithm="neural_net"), 
                        "config_12.04.2022-120420.json"),
                    "boosting": os.path.join(config_path.format(
                        algorithm="boosting"), 
                        "config_16.04.2022-125845.json")
                }
            

        elif self.experiment == "wide_aggregated":
            
            if self.Q == "Q4":
                self.config_names = {
                    "linear_regression": None,
                    "elastic_net": os.path.join(config_path.format( 
                        algorithm="elastic_net"),
                        "config_21.04.2022-084203.json"),
                    "neural_net": os.path.join(config_path.format(
                        algorithm="neural_net"), 
                        "config_18.04.2022-130831.json"),
                    "boosting": os.path.join(config_path.format(
                        algorithm="boosting"), 
                        "config_21.04.2022-084715.json")
                }
            elif self.Q == "Q5":
                self.config_names = {
                    "linear_regression": None,
                    "elastic_net": os.path.join(config_path.format( 
                        algorithm="elastic_net"),
                        "config_21.04.2022-084203.json"),
                    "neural_net": os.path.join(config_path.format(
                        algorithm="neural_net"), 
                        "config_18.04.2022-130831.json"),
                    "boosting": os.path.join(config_path.format(
                        algorithm="boosting"), 
                        "config_21.04.2022-084856.json")
                }
            elif self.Q == "Q6":
                self.config_names = {
                    "linear_regression": None,
                    "elastic_net": os.path.join(config_path.format( 
                        algorithm="elastic_net"),
                        "config_21.04.2022-084203.json"),
                    "neural_net": os.path.join(config_path.format(
                        algorithm="neural_net"), 
                        "config_18.04.2022-130831.json"),
                    "boosting": os.path.join(config_path.format(
                        algorithm="boosting"), 
                        "config_21.04.2022-131239.json")
                }


        elif self.experiment == "wide_item":

            self.config_names = {
                "linear_regression": None,
                "elastic_net": os.path.join(config_path.format( 
                    algorithm="elastic_net"),
                    "config_27.04.2022-184156.json"),
                "neural_net": os.path.join(config_path.format(
                    algorithm="neural_net"), 
                    "config_21.04.2022-190115.json"),
                "boosting": os.path.join(config_path.format(
                    algorithm="boosting"), 
                    "config_22.04.2022-085156.json")
            }

        elif self.experiment == "wide_pca":

            self.config_names = {
                "linear_regression": None,
                "elastic_net": os.path.join(config_path.format( 
                    algorithm="elastic_net"),
                    "config_22.04.2022-095832.json"),
                "neural_net": os.path.join(config_path.format(
                    algorithm="neural_net"), 
                    "config_25.04.2022-185205.json"),
                "boosting": os.path.join(config_path.format(
                    algorithm="boosting"), 
                    "config_27.04.2022-183454.json")
            }

        for algo in algorithms[1:]:

            with open(self.config_names[algo], "r") as f:
                config = json.load(f)
                
            keys = self.hyperparameters[algo]

            if algo == "neural_net":

                keys.remove("input_shape")
                keys.remove("epochs")
                keys.remove("batch_size")


                self.fit_params[algo] = {"epochs": config["epochs"],
                                        "batch_size": config["batch_size"]}
                                        
            else:
                self.fit_params[algo] = None


            self.configs[algo] = {key:config[key] for key in keys}


            if algo == "neural_net":
                if "wide" not in self.experiment:
                    self.configs[algo]["input_shape"] = config["input_shape"][self.Q]
                else:
                    self.configs[algo]["input_shape"] = config["input_shape"]

        self.fit_params["linear_regression"] = None
               
    def get_models(self):

        self.args.config = self.config_names["neural_net"].split("/")[-1]
        self.NN_instance = NeuralNet(self.args)

        def NN_model(input_shape, layer_sizes, lr, dropout=True, l2_reg=False, dropout_rate=None,activation="relu", opt="Adam"):
            def dnn():
                model = self.NN_instance.DNN(input_shape, layer_sizes, lr, dropout, l2_reg, dropout_rate,activation, opt)
                return model
            return dnn

        self.models = {
            "linear_regression": LinearRegression(),
            "elastic_net": ElasticNet(**self.configs["elastic_net"]),
            "boosting": xgb.XGBRegressor(**self.configs["boosting"]),
            "neural_net": KerasRegressor(build_fn=NN_model(**self.configs["neural_net"]))
        }




    def CV(self, n_splits=10):

        self.results = {}

        
        for name, model in self.models.items():
            
            kfold = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=0)
            
            cv_scores = model_selection.cross_val_score(model,
                                                       self.X_train, 
                                                       self.y_train,
                                                       cv=kfold, 
                                                       fit_params=self.fit_params[name],
                                                       scoring=self.metric,
                                                       verbose=1)


            self.results[name] = list(cv_scores)
            # pdb.set_trace()
        self.results["config_names"] = self.config_names

        outfile_path = self.path + "cv_results"

        if not os.path.exists(outfile_path):
            os.mkdir(outfile_path)

        outfile_name =  f"10cv_results_{self.Q}.json"
        with open(os.path.join(outfile_path, outfile_name), "w") as f:
            json.dump(self.results, f, indent=4)

    def five_X_two_CV(self):

        pairs = [
            ("linear_regression", "elastic_net"),
            ("linear_regression", "neural_net"),
            ("linear_regression", "boosting"),
            ("elastic_net", "neural_net"),
            ("elastic_net", "boosting"),
            ("neural_net", "boosting")

        ]

        self.results = []

        count=1
        for est1, est2 in pairs:

            print(f"{count}/{len(pairs)} - Current pair: {est1} & {est2}\n")

            model1 = self.models[est1]
            model2 = self.models[est2]

            fit_params1 = self.fit_params[est1]
            fit_params2 = self.fit_params[est2]
            # pdb.set_trace()
            t_stat, pvalue = paired_ttest_5x2cv(model1,
                                                model2,
                                                fit_params1,
                                                fit_params2,
                                                self.X_train,
                                                self.y_train,
                                                scoring=self.metric,
                                                random_seed=0)

            if pvalue > 0.05:
                print(f"t test statistic: {t_stat:.3f} \np-value: \033[91m{pvalue:.3f}\033[0m \n")
            else:
                print(f"t test statistic: {t_stat:.3f} \np-value: \x1b[92m{pvalue:.3f}\033[0m \n")
            
            result_df = {"estimator1": est1, "estimator2": est2, "t_statistic": t_stat, "pvalue": pvalue}
            self.results.append(result_df)
            count +=1

        self.results.append(self.config_names)

        outfile_path = self.path + "cv_results/" +  f"5x2cv_ttest_results_{self.Q}.json"
        with open(outfile_path, "w") as f:
            json.dump(self.results, f, indent=4)


    def _get_dataframes(self):

        if self.experiment == "wide_aggregated":

            df_train = f"aggregated_train_sub_dataframe_scl_LTHofMD_RSS_ss_edu_imm_abuse_income_GSE_anger_RSES_SWLS_{self.Q_list}.csv"
            df_test = f"aggregated_test_sub_dataframe_scl_LTHofMD_RSS_ss_edu_imm_abuse_income_GSE_anger_RSES_SWLS_{self.Q_list}.csv"

        elif self.experiment == "wide_item":
            
            df_train = f"train_sub_dataframe_scl_LTHofMD_RSS_ss_edu_imm_abuse_income_GSE_anger_RSES_SWLS_{self.Q_list}.csv"
            df_test = f"test_sub_dataframe_scl_LTHofMD_RSS_ss_edu_imm_abuse_income_GSE_anger_RSES_SWLS_{self.Q_list}.csv"

        elif self.experiment == "wide_format" or self.experiment == "wide_pca":

            df_train = f"train_imputed_cleaned_dataframe_{self.Q_list}.csv"
            df_test = f"test_imputed_cleaned_dataframe_{self.Q_list}.csv"

      
        df_path = self.path + "dataframes/"

        self.train_df = pd.read_csv(df_path + df_train)
        self.test_df = pd.read_csv(df_path + df_train)
            



def main():

    args = get_args()

    Q = args.qs[-1].upper()

    CV = CrossValidater(args, Q)
    CV()
    # CV.five_X_two_CV()



if __name__=='__main__':


    main()
