## Folder structure

Each folder, except for ``dataframes``, corresponds with different ways of handling the data.

### delta_Q1_Q4

Here we have looked at the difference in time-dependent variables between Q1 and Q4, thus reducing number of time points to one. 

### Q1_independent_var

Here we have used the data in the wide format, using the scl score at Q4 as the response variable, i. e. turning the scl score at  Q1 and all of the Q1-variables into independent variables. 

### long_format_pred

Here the long format of the data is used. Training the deep learning methods for both of the time points, and thus making two prediction for each person. 
