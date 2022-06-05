## Folder Structure

Here each folder contains numerical experiments performed with the datasets storted inside each folder, specifically in the folder named `dataframes`. However, these folders are currently empty due to privacy regulations.

For the folders `Q1_Q3_QX`, the training data are the prenatal exposures from Q1 and Q3, and the target is the last questionnaire, QX. These folders are creating for the first numerical experiment, while `Q4`, `Q5` and `Q6` contain results for the second experiment.

In the first experiment five different datasets were created, so each `Q1_Q3_QX` folder has a sub-folder for each dataset. The results for the correlated dataset is found together with the complete dataset.

The following structure is found inside the different experiment folders:
```
Numerical Experiment
├── dataframes

├── Q1_Q3_Q{4,5,6}
│   ├── dataframes
│   ├── wide_aggregated_analysis
│       ├── boosting
│           ├── configs
│           ├── feature_importance
│           ├── results
│       ├── dataframes
│       ├── elastic_net
│           ├── coefficients
│           ├── configs
│           ├── results
│       ├── linear_reggression
│           ├── coefficients
│           ├── configs
│           ├── results
│       ├── neural_net
│           ├── configs
│           ├── results
│   ├── [the remaining dataset folders]
|
├── Q{4,5,6}
|   ├── single_analysis
│       ├── boosting
│           ├── [...]
│       ├── dataframes
│       ├── elastic_net
│           ├── [...]
│       ├── linear_reggression
│           ├── [...]
│       ├── neural_net
│           ├── [...]
```
