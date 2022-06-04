import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
import hjson
from sklearn.model_selection import train_test_split

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer


np.random.seed(1)


def get_files(extension, questionnaires):
    """Gives the paths to the given questionnaires

    Args:
        extension: extension of the datafiles, can be 'sav' or 'csv'
        questionnaires: list containing the questionnaire codes

    Returns:
        q_files: list of the filenames with absolute path

    """

    dirpath = "/tsd/p805/data/durable/data/moba/Original files/{}/".format(extension)

    files = os.listdir(dirpath)
    files = [fn for fn in files if ".{}".format(extension) in fn]

    q_files = []

    for i in range(len(questionnaires)):
        for fn in files:
            if questionnaires[i] in fn:
                q_files.append(dirpath + fn)

    return q_files


def get_selected_columns(df, variables, variable_keys, questionnaires):
    """Picks out selected columns from dataframe and place them in a new DataFrame

    Args:
        df: original dataframe contining the columns you want to extract
        variables: nested dict containing specificed variables for selected questionnaires
        variable_keys: list holding the keys to the variables you want to extract
        questionnaires: list containing which questionnaire(s) the variables belong to

    Return:
        new_df: new dataframe containing the columns extracted
    """

    # New dataframe to hold the selected columns
    new_df = pd.DataFrame()

    # Adding the first column from the original dataframe (usually the id column)
    new_df = pd.concat([new_df, df["PREG_ID_2601"].to_frame()])

    # List over keys present in the dataframe
    keys_present = []

    # Iterating through the variables dict
    for Qs, nested_dict in variables.items():

        # Only including the variables from the specified questionnaires
        if Qs in questionnaires:

            # Iterating through the variables for each questionnaire
            for key, values in nested_dict.items():

                # Extracting the columns of interest
                if key in variable_keys:
                    
                    try:
                        new_df[values] = df[values]

                        keys_present.append(key)
                    except KeyError:
                        print(f"Items for {key} not found in questionnaire {Qs} \n")
                        # sys.exit("Exiting")

    return new_df, keys_present


def split_data(df, fraction_train=0.8):
    """
    Splits the data into a training set and a test set
    Training/test is by default 80/20 ratio.

    Args:
        df: pandas dataframe containing both predictors and target
        fraction_train: fraction of the dataset used for training.

    Returns:
        df_train, df_test: new dataframes containing the partitioned data
    """

    # Total number of rows
    N = df.shape[0]

    # Shuffle dataset before partitioning the data
    shuffle_idx = np.random.permutation(N)
    df = df.iloc[shuffle_idx].reset_index(drop=True)

    # Number of training samples
    N_train = int(fraction_train * N)

    # Train dataset
    df_train = df.iloc[:N_train, :]

    # Test dataset
    df_test = df.iloc[N_train:, :]

    return df_train, df_test

def prepare_folds(train_df, test_df, target_col, remove_cols=None, validation=True, remove_nan=False):
    """Function that prepares the dataframes into train, validation and test folds.
    
    The function splits the training data into train and validation folds with a 80/20 ratio. The target column is removed from both
    dataframes and placed in a separate variable.

    Args:
        train_df: Pandas dataframe containing all training data
        test_df: Pandas dataframe containing all test data
        target_col: Name [string] of the target column
        remove_cols: list of strings, columns that should not be included in the returned data
        validation: If True, returns a validation set
        remove_nan: If True, removes all of the rows containing nan-values

    Returns:
        if validatione = True:
            X_train, X_val, X_test, y_train, y_val, y_test: Numpy ndarray, with X being the design matrices and y being the targets
        else: 
            X_train, X_test, y_train, y_test

    """

    if remove_cols is None:
        remove_cols = [target_col]
    elif isinstance(remove_cols, list):
        remove_cols.append(target_col)
    else:
        print(f"remove_cols should be either None or list, not {type(remove_cols)}")


    if remove_nan:
        # Since data is not imputated, for now, removing all rows with nan values
        train_df.dropna(axis=0, inplace=True)
        test_df.dropna(axis=0, inplace=True)

    # Preparing train, validation and test variables
    y_train = train_df[target_col].to_numpy().reshape(-1,1)
    y_test = test_df[target_col].to_numpy().reshape(-1,1)


    X_train = train_df.copy()
    X_test = test_df.copy()


    # Remove unwanted columns
    X_train.drop(columns=remove_cols, inplace=True)
    X_test.drop(columns=remove_cols, inplace=True)
    

    if validation:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=1
        )


        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test


def paired_ttest_5x2cv(estimator1, estimator2, 
                       fit_params1, fit_params2,
                       X, y,
                       scoring=None,
                       random_seed=None):
    """
    Implements the 5x2cv paired t test proposed
    by Dieterrich (1998)
    to compare the performance of two models.
    Parameters
    ----------
    estimator1 : scikit-learn classifier or regressor
    estimator2 : scikit-learn classifier or regressor
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values.
    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.
    random_seed : int or None (default: None)
        Random seed for creating the test/train splits.
    Returns
    ----------
    t : float
        The t-statistic
    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/
    """
    rng = np.random.RandomState(random_seed)

    if scoring is None:
        if estimator1._estimator_type == 'classifier':
            scoring = 'accuracy'
        elif estimator1._estimator_type == 'regressor':
            scoring = 'neg_mean_squared_error'
        else:
            raise AttributeError('Estimator must '
                                 'be a Classifier or Regressor.')
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    variance_sum = 0.
    first_diff = None

    def score_diff(X_1, X_2, y_1, y_2):

        if fit_params1 is None:
            estimator1.fit(X_1, y_1)
        else:
            estimator1.fit(X_1, y_1, **fit_params1)
        
        if fit_params2 is None:
            estimator2.fit(X_1, y_1)
        else:
            estimator2.fit(X_1, y_1, **fit_params2)

        # estimator2.fit(X_1, y_1, fit_params2)
        est1_score = scorer(estimator1, X_2, y_2)
        est2_score = scorer(estimator2, X_2, y_2)
        score_diff = est1_score - est2_score
        return score_diff

    for i in range(5):

        randint = rng.randint(low=0, high=32767)
        X_1, X_2, y_1, y_2 = \
            train_test_split(X, y, test_size=0.5,
                             random_state=randint)

        score_diff_1 = score_diff(X_1, X_2, y_1, y_2)
        score_diff_2 = score_diff(X_2, X_1, y_2, y_1)
        score_mean = (score_diff_1 + score_diff_2) / 2.
        score_var = ((score_diff_1 - score_mean)**2 +
                     (score_diff_2 - score_mean)**2)
        variance_sum += score_var
        if first_diff is None:
            first_diff = score_diff_1

    numerator = first_diff
    denominator = np.sqrt(1/5. * variance_sum)
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), 5)*2.
    return float(t_stat), float(pvalue)
