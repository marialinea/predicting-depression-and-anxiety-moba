import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import pdb


###################### Set Up ###########################

# Which questionnaires are we looking at
# Qs = ["Q1","Q3", "Q4", "Q5", "Q6"]
Qs = ["Q4", "Q5", "Q6"]

Q_names = "_".join(Qs)

# Dataframe structure - should correspond with a subfolder in the experiment folder
# data = "wide_format_analysis"
data = "wide_aggregated_analysis"

# Name of dataframe file containing true targets
test_df_name = "test_imputed_cleaned_dataframe_Q1_Q3_{Q}.csv"
train_df_name = "train_imputed_cleaned_dataframe_Q1_Q3_{Q}.csv"

# Target preprocessing
target_preprocessing = "mean"  # "mean" or "sum"

# Target column
target_col = f"{target_preprocessing}_scl"

# Name of the file to store the predictions
outfile_name = "distribution_preds_true_wide_aggregated.pdf"


###################### Load Predictions ################################

predictions = {}

# preds_name = {"linear_regression": ["preds_true_12.04.2022-114130.npy", "preds_true_12.04.2022-122143.npy", "preds_true_13.04.2022-094536.npy"],
#                "elastic_net": "preds_true_19.04.2022-175857.npy",#"preds_true_12.04.2022-114232.npy",
#                 "neural_net": "preds_true_12.04.2022-120420.npy",
#                "boosting":  "preds_true_16.04.2022-125845.npy"
#               }

preds_name = {"linear_regression": ["preds_true_19.04.2022-190154.npy", "preds_true_19.04.2022-191927.npy", "preds_true_19.04.2022-191942.npy"],
               "elastic_net": "preds_true_21.04.2022-084203.npy",
                "neural_net": "preds_true_18.04.2022-130831.npy",
               "boosting":  ["preds_true_21.04.2022-084715.npy", "preds_true_21.04.2022-084856.npy", "preds_true_21.04.2022-131239.npy"]
              }



# preds_name = {"linear_regression": "preds_true_{Q}_06.03.2022-100223.npy",
#                "elastic_net": "preds_true_{Q}_06.03.2022-095645.npy",
#                 "neural_net": "preds_true_{Q}_06.03.2022-100500.npy",
#                "boosting": "preds_true_{Q}_06.03.2022-100600.npy",
#               }

# Venter på at boosting og nevralt nett skal ha prediksjoner klare

methods =  list(preds_name.keys())



for i, Q in enumerate(Qs):

    predictions[Q] = {}

    for method in methods:

        if method == "linear_regression" or method=="boosting":
            path = os.path.join(f"../experiments/Q1_Q3_{Q}/{data}/{method}/results/{preds_name[method][i]}")
        else: 
            path = os.path.join(f"../experiments/Q1_Q3_{Q}/{data}/{method}/results/{preds_name[method]}")

        predictions[Q][method] = np.load(path)
       



########################## Make plot #################################


# sns.set_palette("pastel")
# plt.rc("text", usetex=True)
# fig, ax = plt.subplots(nrows=2,ncols=3, sharex= True, sharey=True)


# fig = plt.figure()
fig, ax = plt.subplots(nrows=4, ncols=3, sharey=True, sharex = True, figsize=(12,9))
# fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(9,5))
# fig.delaxes(ax[0,2]) 
cmap = plt.cm.get_cmap('jet')

ax_dict = {
    "linear_regression": {"Q4": ax[0,0], "Q5": ax[0,1], "Q6": ax[0,2]},
    "elastic_net": {"Q4": ax[1,0], "Q5": ax[1,1], "Q6": ax[1,2]},
    "neural_net": {"Q4": ax[2,0], "Q5": ax[2,1], "Q6": ax[2,2]},
    "boosting": {"Q4": ax[3,0], "Q5": ax[3,1], "Q6": ax[3,2]}
}

# ax_dict = {"Q4": ax[0],
#            "Q5": ax[1],
#            "Q6": ax[2]}

# ax_dict = {"Q1": ax[0,0],
#            "Q3": ax[0,1],
#            "Q4": ax[1,0],
#            "Q5": ax[1,1],
#            "Q6": ax[1,2]}



colors = [0.15, 0.35, 0.65, 0.85]
label_dict = {"linear_regression": "Linear Regression",
               "elastic_net": "Elastic Net",
                "boosting": "XGBoost",
                "neural_net": "Neural Net",
                "true": "True Targets"
}
FONTSIZE = 13

#########################################################################
# true_targets = {}

# minority = {"train": [],
#             "test": []}

# for Q in Qs:
#     path = f"../experiments/Q1_Q3_{Q}/{data}/dataframes/"

#     train_df = pd.read_csv(path+train_df_name.format(Q=Q))
#     test_df = pd.read_csv(path+test_df_name.format(Q=Q))

#     above_train = np.where(train_df[f"mean_scl_{Q}"]  > 1.75)[0]
#     above_test = np.where(test_df[f"mean_scl_{Q}"] > 1.75)[0]

#     minority["train"].append(len(above_train)/train_df.shape[0] * 100)
#     minority["test"].append(len(above_test)/test_df.shape[0] * 100)

#     true_target = np.concatenate([train_df[f"mean_scl_{Q}"], test_df[f"mean_scl_{Q}"]])

#     true_targets[Q] = true_target

# print(minority)

# for Q, items in true_targets.items():
#     sns.histplot(items, label="True", fill=True, color=cmap(colors[0]),alpha=0.3,binwidth=0.125, ax=ax_dict[Q])
#     ax_dict[Q].set_title(f"{Q}", fontsize=FONTSIZE) 
    

# ax[0].set_ylabel("Distribution of True Targets", fontsize=FONTSIZE)
# ax[1].set_xlabel(r"Mean SCL Scores In $\mathcal{D}$", fontsize=FONTSIZE)
# plt.savefig(f"./../figures/distribution_target_whole_dataset.pdf")
#########################################################################


for i, method in enumerate(methods):
    for j, Q in enumerate(Qs):
        if method == "linear_regression":
            # sns.kdeplot(predictions[Q][method][:,1], label="True",fill=True, alpha=0.2, color="c",linestyle="--", ax=ax_dict[Q])
            ax_dict[method][Q].set_title(f"{Q}", fontsize=FONTSIZE) 
            
       
        sns.histplot(predictions[Q][method][:,1], label="True Targets", fill=False, color="k",binwidth=0.125, ax=ax_dict[method][Q])
        sns.histplot(predictions[Q][method][:,0], label=label_dict[method], fill=True, alpha=0.5, color=cmap(colors[i]), binwidth=0.125, ax=ax_dict[method][Q])
        # sns.kdeplot(predictions[Q][method][:,0], label=label_dict[method],fill=False, alpha=0.5,color=cmap(colors[i]), ax=ax_dict[Q])
        ax_dict[method][Q].set_xlabel("")
        ax_dict[method][Q].set_ylabel("")
        if (j+1)%3 == 0:
            ax_dict[method][Q].legend()


# for Q in Qs:
#     for i, method in enumerate(methods):
#         if method == "linear_regression":
#             # sns.histplot(predictions[Q][method][:,1], label="True", fill=False, color="k",binwidth=0.125, ax=ax_dict[Q])
#             sns.kdeplot(predictions[Q][method][:,1], label="True",fill=True, alpha=0.2, color="c",linestyle="--", ax=ax_dict[Q])
            
        
#         # sns.histplot(predictions[Q][method][:,0], label=label_dict[method], fill=True, alpha=0.5, binwidth=0.125, color=cmap(colors[i]), ax=ax_dict[Q])
#         sns.kdeplot(predictions[Q][method][:,0], label=label_dict[method],fill=False, alpha=0.5,color=cmap(colors[i]), ax=ax_dict[Q])
#         ax_dict[Q].set_xlabel("")
#         ax_dict[Q].set_ylabel("")
#         ax_dict[Q].set_title(f"{Q}", fontsize=FONTSIZE) 



# lines, labels = ax[3,2].get_legend_handles_labels()
# pretty_labels = [label_dict[label] for label in labels]
# fig.legend(lines[0], ["True Targets"], loc="upper center", bbox_to_anchor=(0.8, 0.8), fancybox=True, fontsize=FONTSIZE)
# fig.xticks(fontsize=FONTSIZE)
# fig.yticks(fontsize=FONTSIZE)


fig.text(0.05, 0.35, "Distribution of Predictions", ha="center", rotation="vertical", fontsize=FONTSIZE)
fig.text(0.5, 0.02, r"Mean SCL Scores In $\mathcal{D}_{test}$", ha="center", fontsize=FONTSIZE)
plt.xlim(0.7,4)
plt.savefig(f"./../figures/{outfile_name}")











