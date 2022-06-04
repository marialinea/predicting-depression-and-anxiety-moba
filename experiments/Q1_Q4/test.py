import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# test_df = pd.read_csv("./dataframes/mean_scl/long_test_preprocessed_sub_dataframe_mean_scl_LTHofMD_RSS_ss_edu_imm_income_SWLS_ALE_Q1_Q4.csv")

# lm_Q1 = []
# lm_Q4 = []
# lm_all = []

# models = ["3", "4", "5"]

# for model in models:

#     lm_all_model = []
#     with open("./Q1_independent_var/results/mean_scl/pred_model{}.txt".format(model), "r") as infile:
#         lines = infile.readlines()
#         for line in lines:
#             preds = line.split()
#             for pred in preds:
#                 try:
#                     lm_all_model.append(float(pred))
#                 except ValueError:
#                     continue
#     lm_all.append(lm_all_model)

#     Q1_model = []
#     Q4_model = []
#     for Q1, Q4 in zip(lm_all_model[::2], lm_all_model[1::2]):
#         Q1_model.append(Q1)
#         Q4_model.append(Q4)

#     lm_Q1.append(Q1_model)
#     lm_Q4.append(Q4_model)

# true = test_df["mean_scl"]
# true = true[true >= 1]
# #true = true.dropna()


# for i in range(3):
#     print(f"For model {i}, min value =", min(lm_Q4[i]))

# print("True model min = ", true.min())


# #sns.histplot(true, kde ="True", label="True")

# # sns.displot(true, kind="kde")
# sns.kdeplot(true, label="True")
# for i, model in enumerate(models):
#     sns.kdeplot(lm_Q4[i], label="Model {}".format(model))
# plt.legend()
# plt.savefig("./Q1_independent_var/results/mean_scl/lm_model_selection.pdf")

true_preds = pd.read_csv("./dataframes/mean_scl/test_preprocessed_sub_dataframe_mean_scl_LTHofMD_RSS_ss_edu_imm_income_SWLS_ALE_Q1_Q4.csv")
true_preds = true_preds["mean_scl_Q4"]
xgb_squared = np.load("./Q1_independent_var/results/mean_scl/boosting_pred_squarederror.npy")
xgb_logsquared = np.load("./Q1_independent_var/results/mean_scl/boosting_pred_squaredlogerror.npy")

sns.histplot(true_preds,  binwidth=0.1, label="True", color="grey", kde=True)
sns.histplot(xgb_squared, binwidth=0.1,  label="Squared Error", color="violet", alpha=0.5, kde=True)
sns.histplot(xgb_logsquared, binwidth=0.1,  label="Squared Log Error", color="lawngreen", alpha=0.5, kde=True)
plt.legend()
plt.savefig("./Q1_independent_var/figures/histplot_xgb_squared_vs_logsquared.pdf")