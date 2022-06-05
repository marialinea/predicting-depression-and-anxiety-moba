import pandas as pd
import numpy as np
import os
import datetime
import argparse
import re
import pdb




def get_args():
    argparser = argparse.ArgumentParser(description="")

    argparser.add_argument(
        "-qs",
        type=str,
        nargs="+",
        default=["Q1", "Q3", "Q4", "Q5", "Q6"],
        choices= ["q1","Q1","q3","Q3", "q4","Q4","q5","Q5","q6","Q6"],
        help="A model is trained for each questionnaire specified."
             "The questionnaires also specify where the input- and output data is retrived and  stored."
    )

    argparser.add_argument(
        "-e",
        type=str,
        nargs="?",
        choices= ["wide_format", "wide_pca",
                 "wide_aggregated", "wide_item", "single"],
        required=True,
        help="Specifies experiment type, i.e., which dataset to use as training and test sets."
    )

    argparser.add_argument(
        "-m",
        type=str,
        nargs="?",
        choices=["tune", "train", "train_eval", "eval"],
        default="train",
        help="If tune, the models hyperparameters are tuned. If train, a config_UNIQUE-ID.json file should be in the model_path to be loaded in, and the model is then trained."
    )

    argparser.add_argument(
        "-c",
        type=str,
        nargs="?",
        default=None,
        help="Name of config file if model is already tuned. If a config name is passed when mode=tune, config name is set to None. Accept partial name of config, as long as the partial name is unique for the config file in the config directory."
    )


    argparser.add_argument(
        "-corr",
        type=str,
        nargs="?",
        choices=["true", "false", "True", "False", "t", "f"],
        default="False",
        help="If True, only features with high correlation with the target variable are included"
    )



    args = argparser.parse_args()

    return args



class Configure(object):
    """

    Args:
        args: argparse object
        algortihm: str,  supervised learning algorithm [linear_regression, elastic_net, neural_net]
    """

    def __init__(self, args, algorithm):

        self.args = args
        self.algorithm = algorithm

        accepted_algorithms = ["linear_regression", "elastic_net", "neural_net", "boosting"]
        if self.algorithm not in accepted_algorithms:
            raise ValueError(f"{self.algorithm} not recognized, must be one of the following: {accepted_algorithms}")



        self._set_up()


    def _set_up(self):
        """
        Function to set up all the necessary variables for all the different
        experiments. These variables are needed regardless of experiment type.
        """
        args = self.args

        # Questionnaires
        self.Q_names = [q.upper() if q.islower() else q for q in args.qs]

        self.Q_list = "_".join(self.Q_names)

        # Tune or train mode
        if self.algorithm != "linear_regression":
            self.mode = args.mode


        # Experiment type
        self.experiment = args.experiment


        # Root directory for current experiment
        self.root = f"./experiments/{self.Q_list}/{self.experiment}_analysis"

        if args.config is not None:

            self.config_path = os.path.join(self.root, self.algorithm, "configs")

            self._get_config()

        else:

            self.unique_id = datetime.datetime.now().strftime("%d.%m.%Y-%H%M%S")

            self.config_name = f"config_{self.unique_id}.json"
            self.args.config = self.config_name

        # Target preprocessing procedure
        self.target = args.target

        self.target_column = f"{self.target}_scl_{self.Q_names[-1]}"

        # Specify columns to remove when preparing train and test data

        self.remove_cols = ["PREG_ID_2601"]

        if "pca" in self.experiment:
            self.PCA = True
        else:
            self.PCA = False


        self._check_dir(self.root)

        self._get_directories()

        self._get_dataframes()

        self._check_corr()

    def _get_directories(self):
        """
        Helper function to get relative paths to the dataframes, result and model directories.
        """

        root = self.root
        algo = self.algorithm

        # Path to experiment folder, i. e. different combinations of questionnai
        self.data_path = os.path.join(root, "dataframes")

        # Path to specific algorithm directory
        root_algo = os.path.join(root, algo)

        self._check_dir(root_algo)

        # Path to result folder
        self.result_path = os.path.join(root, algo, "results")

        self._check_dir(self.result_path)

        if self.algorithm == "linear_regression" or self.algorithm == "elastic_net":

            # Path to regression coefficients
            self.coeff_path = os.path.join(root, algo, "coefficients")

            self._check_dir(self.coeff_path)
        else:

            # Path to models
            self.model_path = os.path.join(root, algo, "models")

            self._check_dir(self.model_path)

        if self.algorithm == "boosting":

            self.feature_imp_path = os.path.join(root, algo, "feature_importance")

            self._check_dir(self.feature_imp_path)

            self.unique_feature_path = os.path.join(self.feature_imp_path, self.unique_id)

            self._check_dir(self.unique_feature_path)

    def _get_dataframes(self):
        """
        Helper function to load the dataframes belonging to the specified experiment.
        All dataframes are stored in the train_test_dict.
        """

        # List containing the filenames to all of the questionnaires
        files = os.listdir(self.data_path)

        # Dictionary to hold the train and test dataframes for the different questionnaires
        self.train_test_dict = {}


        for fn in files:

            if "train" in fn:
                # Loading train dataframe
                self.train_test_dict["train"] =  pd.read_csv(os.path.join(self.data_path, fn))

            elif "test" in fn:

                # Loading test dataframe
                self.train_test_dict["test"] =  pd.read_csv(os.path.join(self.data_path, fn))

    def _get_config(self):


        files = os.listdir(self.config_path)
        counter = 0
        for fn in files:
            if fn.startswith(self.args.config) is True:

                words = re.split("\.|_", fn)

                self.config_name = fn

                self.args.config = self.config_name

                self.unique_id = ".".join(words[1:-1])

                counter +=1

        if counter < 1:
            print(f"Could not find {self.args.config} in {self.config_path}")
            raise FileNotFoundError
        elif counter > 1:
            print(f"Found {counter} files with the name {self.args.config}")
            raise ValueError

    def _check_dir(self, directory):
        """
        Function that checks if a directory exists, and if not creates it.
        """

        if not os.path.exists(directory):
            os.mkdir(directory)

    def _check_corr(self):

        if "wide" in self.experiment:

            args = self.args

            if args.correlation == "t" or args.correlation == "true" or args.correlation == "True":

                train_df = self.train_test_dict["train"]
                test_df = self.train_test_dict["test"]

                corr = train_df.corr()

                target_corr = np.abs(corr[self.target_column])
                high_corr = np.where(target_corr > 0.35)[0]
                high_cols = train_df.columns[high_corr]

                train_df = train_df[high_cols]
                test_df = test_df[high_cols]

                self.train_test_dict["train"] = train_df
                self.train_test_dict["test"] = test_df

                self.unique_id = self.unique_id + "-high_corr"
