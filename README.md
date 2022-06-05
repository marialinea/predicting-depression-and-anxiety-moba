# A machine learning approach to understanding depression and anxiety in new mothers
### Predicting symptom levels using population-based registry data from a large Norwegian prospective study

This repository contain all code written to produce the results found in ADD LINK HERE!
We investigated the ability of machine learning methods to predict a continuous measure of symptoms of anxiety and depression in new mothers, using data from a large population-representative prospective cohort. The data is not publicly available due to privacy regulations, so all of the folders dedicated to storing the data are empty.

This repository has the following structure

```
Master Thesis Project
├── data
│   ├── ids
│   ├── all_variables.json
│   ├── selected_variables.json
|
├── experiments
│   ├── dataframes
│   ├── [several experiment folders]
|   
├── figures
├── scripts
├── utils
|
├── cross_validater.py
├── main_en.py
├── main_lr.py
├── main_nn.py
├── main_xgb.py
├── prepare_data.py

```

## Processing Data
---------------------------
In the `scripts` folder, there are scripts for constructing a base dataset from the raw files, and scripts to process results. See README file inside folder to get an overview over all of the scripts.

The `prepare_data.py` scripts will clean, split the data into train/test splits and impute the data.

## Make Predictions
----------------------------------------
All of the scripts in the

All of the scripts in the main folder, except `prepare_data.py`, are used to make predictions. The `main_XXX.py` scrips run either a linear regression model (lr), an elastic net (en), a neural network (nn) or a XGBoost model (xgb). The `cross_validater.py` script does not produce useful results for the multiple linear regression model.

They take the following command line arguments
 ```  
  -h, --help            show this help message and exit

  -qs [{q1,Q1,q3,Q3,q4,Q4,q5,Q5,q6,Q6}]
                        A model is trained for each questionnaire specified.The questionnaires
                        also specify where the input- and output data is retrived and stored.

  -e [{wide_format, wide_pca, wide_aggregated, wide_item,single}]
                        Specifies experiment type, i.e., which dataset to use as training and test sets.

  -m [{tune,train,train_eval,eval}]
                        If tune, the models hyperparameters are tuned. If train,
                        a config_UNIQUE-ID.json file should be in the model_path to
                        be loaded in, and the models is then trained.

    -c [C]              Name of config file if model is already tuned. If a config name is passed when
                        mode=tune, config name is set to None. Accept partial name of config, as long as
                        the partial name is unique for the config file in the config directory.

  -corr [{true,false,True,False,t,f}]
                        If True, only features with high correlation with the target variable are included

 ```
