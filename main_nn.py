import numpy as np
import pandas as pd
import  json, datetime, itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

import os, sys, re, pdb

from utils.configure import Configure, get_args
from utils.data import prepare_folds
from utils.print import print_info


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import backend as K

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils.fixes import loguniform
from sklearn.utils import shuffle
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA


from sklearn.model_selection import train_test_split


class NeuralNet(Configure):
    def __init__(self, args, algorithm="neural_net"):
        super().__init__(args, algorithm)
        
        
        self._nn_config()

    def _nn_config(self):
        """
        Helper function to configure the specific neural network class
        """
        
        self.log_path = os.path.join(self.root, self.algorithm, "logs")
        
        self._check_dir(self.log_path)
        
        self.config_path = os.path.join(self.root, self.algorithm, "configs")
        
        self._check_dir(self.config_path)
        
    
    def DNN(self, input_shape, layer_sizes, lr, dropout=True, l2_reg=False, dropout_rate=None,activation="relu", opt="Adam"): 
            """
            A deep neural network with possbile drop-out layers
            """

            models = keras.models
            layers = keras.layers
            
            opt_dict = {
                "Adam" : keras.optimizers.Adam,
                "SGD" : keras.optimizers.SGD
            }


            model = models.Sequential()
            


            # Input layer
            if dropout is True:
                model.add(layers.Dropout(0.8, input_shape=(input_shape,),name="input"))
            else: 
                model.add(keras.Input(shape=(input_shape,), name="input"))

            # Hidden layers
            layer_count = 1
            for i, layer_size in enumerate(layer_sizes, start=1):

                if dropout is True:

                    # Add Dropput layer every other layer
                    if i%2==0:
                        model.add(layers.Dropout(dropout_rate, name=f"hidden_layer{layer_count}"))
                        layer_count +=1

                if l2_reg is False:
                    model.add(layers.Dense(layer_size, 
                                        activation=activation,
                                        kernel_constraint=MaxNorm(3),
                                        name=f"hidden_layer{layer_count}"))
                    layer_count += 1
                else:
                    model.add(layers.Dense(layer_size, 
                                        activation=activation,
                                        kernel_regularizer="l2",
                                        name=f"hidden_layer{layer_count}"))
                    layer_count += 1

            # Output layer
            model.add(layers.Dense(1, activation="linear", name="output"))

            def root_mean_squared_error(y_true, y_pred):
                return K.sqrt(K.mean(K.square(y_pred-y_true)))

            model.compile(
                optimizer=opt_dict[opt](lr),
                loss=root_mean_squared_error,#"mse",
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

            return model

    def tune(self):
        """
        Tune network architecture with RandomizedSearchCV from scikit-learn.
        The results are stored in a json-file and saved in the config directory.
        """
        print("")

        print_info("Tuning Network \n")


        train_df = self.train_test_dict["train"]
        test_df = self.train_test_dict["test"]


        X_train, X_test, y_train, y_test = prepare_folds(
            train_df,
            test_df,
            (self.target_column),
            remove_cols=(self.remove_cols),
            validation=False,
            remove_nan=False,
        )

        X_train = StandardScaler().fit_transform(X_train)

      
        input_shape = X_train.shape[1]

        
        sizes = [10, 100, 500, 1000]
        shapes = (
            list(itertools.product(sizes, repeat=1))
            + list(itertools.product(sizes, repeat=2))
            + list(itertools.product(sizes, repeat=3))
            + list(itertools.product(sizes, repeat=4))
            + list(itertools.product(sizes, repeat=5))
            + list(itertools.product(sizes, repeat=6))
        )
        n_iter = 10

            

        if self.PCA is True:
        
            pca = PCA(n_components=0.95)

            X_train = pca.fit_transform(X_train)

            input_shape = X_train.shape[1]
           

        param_grid = dict(
            input_shape=[input_shape],
            layer_sizes=shapes,
            lr=[0.001, 0.01],
            dropout_rate=[0.2, 0.5],
            batch_size=[32, 64],
            l2_reg=[False, True],
            dropout=[False, True],
            activation=["relu", "tanh"],
            epochs=[100, 500],#, 1000],
            opt=["SGD", "Adam"],
        )


        model = KerasRegressor(build_fn=(self.DNN), verbose=1)

        scoring = "neg_root_mean_squared_error"

        grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=scoring,
            cv=5,
            n_iter=n_iter,
            verbose=2,
        )

        grid_result = grid.fit(X_train, y_train)
        config_dict = grid.best_params_



        config_dict["model_filepath"] = os.path.join(self.model_path, f"models_{self.unique_id}", "model.ckpt")
        config_dict["input_shape"] = input_shape


        with open(f"{self.config_path}/{self.config_name}", "w") as (outfile):
            json.dump(config_dict, outfile, indent=4)
        
        

    def train(self):

        with open(f"{self.config_path}/{self.config_name}", "r") as (infile):
            self.config = json.load(infile)

        
        print_info("\nStart Training Networks with the following parameters: \n")

        print_info(json.dumps((self.config), indent=4))
        

        train_df = self.train_test_dict["train"]
        test_df = self.train_test_dict["test"]


        ########################################
        # cal_housing = pd.read_csv("california.csv")
    
        # y = cal_housing["median_house_value"].to_numpy()
        
        # cal_housing = cal_housing.drop(columns=["ocean_proximity", "median_house_value", "total_bedrooms"])
        # X = cal_housing.to_numpy()


        # X_train, X_test, y_train, y_test = train_test_split(
        #                                 X, y, test_size=0.1, random_state=13
        #                             )
        # X_train, X_val, y_train, y_val = train_test_split(
        #                                 X_train, y_train, test_size=0.1, random_state=13
        #                             )

        # train_df = [X_train, y_train, X_val, y_val]
        # test_df = [X_test, y_test]
        ########################################


        self.train_one(train_df, test_df)

    def train_one(self, train_df, test_df):

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

       
        model = self.get_model()

    
        log_dir = os.path.join(self.log_path, f"logs_{self.unique_id}")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

        model_filepath = self.config["model_filepath"]


        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                filepath=model_filepath,
                                                                monitor="val_loss",
                                                                save_best_only=True,
                                                                save_weights_only=True,
                                                                mode="min",
        )


        history = model.fit(
            X_train,
            y_train,
            batch_size=(self.config["batch_size"]),
            epochs=(self.config["epochs"]),
            validation_data=(X_val, y_val),
            callbacks=[tensorboard_callback, checkpoint_callback],
        )
  

    def eval(self, save_preds=False):

        with open(f"{self.config_path}/{self.config_name}", "r") as (infile):
            self.config = json.load(infile)

        print("")


        print_info("Evaluating Model \n")
        keys = [
            "layer_sizes",
            "opt",
            "lr",
            "l2_reg",
            "activation",
            "dropout",
            "dropout_rate",
        ]

        for key in keys:
            print_info(f"'{key}' : {self.config[key]}")

        results = {}

        train_df = self.train_test_dict["train"]
        test_df = self.train_test_dict["test"]

        ######################################################
        # cal_housing = pd.read_csv("california.csv")
        # y = cal_housing["median_house_value"].to_numpy()
        # cal_housing = cal_housing.drop(columns=["ocean_proximity", "median_house_value","total_bedrooms"])
        # X = cal_housing.to_numpy()

        # X_train, X_test, y_train, y_test = train_test_split(
        #                                 X, y, test_size=0.1, random_state=13
        #                             )

        # train_df = [X_train, y_train]
        # test_df = [X_test, y_test]
        ######################################################
        
        mse, rmse, mae, residuals = self.evaluate_one(train_df, test_df, save_preds=save_preds)

        
        results["mse"] = mse
        results["rmse"] = rmse
        results["mae"] = mae
         
        with open(f"{self.result_path}/nn_metrics_{self.unique_id}.json", "w") as (outfile):
            json.dump(results, outfile, indent=4)

    def evaluate_one(self, train_df, test_df, save_preds=False):
        
        model = self.get_model()

        model_filepath = self.config["model_filepath"]
        result_name = f"preds_true_{self.unique_id}.npy"
        input_shape = self.config["input_shape"]

        checkpoint_dir = os.path.dirname(model_filepath)

        model.load_weights(model_filepath)
     
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

            pca = PCA(n_components=input_shape, svd_solver="arpack", random_state=42)

            X_train = pca.fit_transform(X_train)

            X_test = pca.transform(X_test)

        #########################
        # import shap

        # background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        # e = shap.DeepExplainer(model, X_train)
        # shap_values = e.shap_values(X_test)
        # mean_shap_values = np.mean(shap_values, axis=0)
        
        # np.save(os.path.join(self.result_path, "mean_shap_values.npy"), mean_shap_values)
        # pdb.set_trace()
        #########################


        preds = model.predict(X_test)

        preds_true = np.column_stack((preds, y_test))

        if save_preds is True:
            np.save(os.path.join(self.result_path, result_name), preds_true)

        mse = mean_squared_error(preds, y_test)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - preds))
        residuals = y_test - preds


        return (mse, rmse, mae, residuals)

    def get_model(self):

        config = self.config
        
   
        input_shape = config["input_shape"]
        
        layer_sizes = config["layer_sizes"]
        lr = config["lr"]
        activation = config["activation"]
        opt = config["opt"]
        l2_reg = config["l2_reg"]
        dropout = config["dropout"]

        if dropout is False:
            rate = None
        else:
            rate = config["dropout_rate"]


        model = self.DNN(
            input_shape,
            layer_sizes,
            lr=lr,
            dropout=dropout,
            l2_reg=l2_reg,
            dropout_rate=rate,
            activation=activation,
            opt=opt,
        )
        return model



def main():

    save_preds = True

    args = get_args()

    if args.mode == "tune":

        if args.config is not None:
            args.config = None

        NN = NeuralNet(args)

        NN.tune()
        NN.train()
        NN.eval(save_preds=save_preds)

    elif args.mode == "train" or args.mode == "train_eval":
        
        NN = NeuralNet(args)

        NN.train()

        if args.mode == "train_eval":
            NN.eval(save_preds=save_preds)

    elif args.mode == "eval":
        
        NN = NeuralNet(args)

        if args.config is None:
            raise FileNotFoundError("If mode=eval, config must be specified")

        NN.eval(save_preds=save_preds)
    else:
        raise ValueError



if __name__ == "__main__":
    
    main()
