import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import pdb

from sklearn.metrics import confusion_matrix


###################### Set Up ###########################

# Which questionnaires are we looking at
# Qs = ["Q1","Q3", "Q4", "Q5", "Q6"]
Qs = ["Q4", "Q5", "Q6"]

Q_names = "_".join(Qs)



# Name of dataframe file containing true targets
test_df_name = "test_imputed_cleaned_dataframe_Q1_Q3_{Q}.csv"
train_df_name = "train_imputed_cleaned_dataframe_Q1_Q3_{Q}.csv"

# Target preprocessing
target_preprocessing = "mean"  

# Target column
target_col = f"{target_preprocessing}_scl"

# Name of the file to store the predictions
outfile_name = "cum_dist_residuals_train_all_pca.pdf"


###################### Load Predictions ################################



# residuals_wide = {"linear_regression": ["lr_residuals_29.04.2022-183338.npy", "lr_residuals_29.04.2022-183351.npy", "lr_residuals_29.04.2022-183358.npy"],
#                "elastic_net": "en_residuals_19.04.2022-175857.npy",
#                 "neural_net": "nn_residuals_12.04.2022-120420.npy",
#                "boosting":  "xgb_residuals_16.04.2022-125845.npy"
#               }

# residuals_pca = {"linear_regression": ["lr_residuals_29.04.2022-183434.npy", "lr_residuals_29.04.2022-183413.npy", "lr_residuals_29.04.2022-183425.npy"],
#                "elastic_net": "en_residuals_22.04.2022-095832.npy",
#                 "neural_net": "nn_residuals_25.04.2022-185205.npy",
#                "boosting":  "xgb_residuals_27.04.2022-183454.npy"
#               }

residuals_wide = {"linear_regression": ["preds_true_04.05.2022-194715.npy", "preds_true_04.05.2022-194807.npy", "preds_true_04.05.2022-194828.npy"],
               "elastic_net": "preds_true_19.04.2022-175857.npy",
                "neural_net": "preds_true_12.04.2022-120420.npy",
               "boosting":  "preds_true_16.04.2022-125845.npy"
              }

residuals_pca = {"linear_regression": ["preds_true_04.05.2022-195233.npy", "preds_true_04.05.2022-195207.npy", "preds_true_04.05.2022-195142.npy"],
               "elastic_net": "preds_true_22.04.2022-095832.npy",
                "neural_net": "preds_true_25.04.2022-185205.npy",
               "boosting":  "preds_true_27.04.2022-183454.npy"
              }

residuals_corr = {"linear_regression": ["lr_residuals_29.04.2022-195915-high_corr.npy", "lr_residuals_29.04.2022-195959-high_corr.npy", "lr_residuals_29.04.2022-213937-high_corr.npy"],
               "elastic_net": "en_residuals_29.04.2022-141045-high_corr.npy",
                "neural_net": "nn_residuals_28.04.2022-181748-high_corr.npy",
               "boosting":  "xgb_residuals_28.04.2022-160225-high_corr.npy"
              }


all_residuals = {"format": residuals_wide, "pca": residuals_pca}#, "corr": residuals_corr} #"aggregated": residuals_agg, "item": residuals_item, }

#######################Load True Targets################################

# targets = {}
# for Q in Qs:
#     targets[Q] = {}
#     for dataset in all_residuals.keys():
#         if dataset == "corr":
#             dataset = "format"
#         path = f"../experiments/Q1_Q3_{Q}/wide_{dataset}_analysis/dataframes/test_imputed_cleaned_dataframe_Q1_Q3_{Q}.csv"

#         target = f"mean_scl_{Q}"

#         df = pd.read_csv(path)

#         targets[Q][dataset] = df[target]

###########################################################################

methods =  list(residuals_wide.keys())
predictions = {}
dataset_dict = {"format": "format", "pca": "pca", "corr":"format"}

confusion_list = []
for dataset, residuals in all_residuals.items():

    predictions[dataset] = {}
    
    confusion_vals = np.zeros([4,2])

    for i, Q in enumerate(Qs):

        predictions[dataset][Q] = {}

        sens_spes = np.zeros([4,2])

        for j, method in enumerate(methods):
            
            try:
                if isinstance(residuals[method], list) is True:
                    path = os.path.join(f"../experiments/Q1_Q3_{Q}/wide_{dataset_dict[dataset]}_analysis/{method}/results/{residuals[method][i]}")
                else: 
                    path = os.path.join(f"../experiments/Q1_Q3_{Q}/wide_{dataset_dict[dataset]}_analysis/{method}/results/{residuals[method]}")

                predictions[dataset][Q][method] = np.load(path)

                preds_true = predictions[dataset][Q][method]
               
                high_preds = (preds_true[:,0] > 1.85).astype("int")
                high_true = (preds_true[:,1] > 1.85).astype("int")

                tn, fp, fn, tp = confusion_matrix(high_true, high_preds).ravel()

                sens_spes[j,0] = tp/(tp + fn) 
                sens_spes[j,1] = tn/(tn + fp) 

            except ValueError:
                pdb.set_trace()

        confusion_vals += sens_spes
        
    

    confusion_list.append(confusion_vals/len(Qs))


#############################################################################

pdb.set_trace()

"""

fig, ax = plt.subplots(nrows=4, ncols=3, sharey=True, sharex = True, figsize=(12,9))
 
cmap = plt.cm.get_cmap("jet")

ax_dict = {
    "linear_regression": {"Q4": ax[0,0], "Q5": ax[0,1], "Q6": ax[0,2]},
    "elastic_net": {"Q4": ax[1,0], "Q5": ax[1,1], "Q6": ax[1,2]},
    "neural_net": {"Q4": ax[2,0], "Q5": ax[2,1], "Q6": ax[2,2]},
    "boosting": {"Q4": ax[3,0], "Q5": ax[3,1], "Q6": ax[3,2]}
}



colors = [0.15, 0.55, 0.75, 0.55]

label_dict = {"linear_regression": "Linear Regression",
               "elastic_net": "Elastic Net",
                "boosting": "XGBoost",
                "neural_net": "Neural Net",
                "true": "True Targets"
}

legend_dict = {"aggregated": "Subselection w/Aggregation",
                "item": "Subselection wo/Aggregation",
                "format": "All Available Items",
                "pca": "Principal Components",
                "corr": "Correlated Features"}
FONTSIZE = 13


for i, (dataset, residuals) in enumerate(predictions.items()):

    for j, method in enumerate(methods):

        for k, Q in enumerate(Qs):

          
            # true_target = targets[Q][dataset_dict[dataset]]

            if method == "linear_regression":
                ax_dict[method][Q].set_title(f"{Q}", fontsize=FONTSIZE) 
                
            
            preds = residuals[Q][method][:,0]
            true = residuals[Q][method][:,1]
            res = true - preds
            # sns.histplot(residuals[Q][method][:,1], label="True Targets", fill=True, color="k",alpha=0.05,binwidth=0.125, ax=ax_dict[method][Q])
            # sns.histplot(residuals[Q][method][:,0], label=legend_dict[dataset], fill=False, color=cmap(colors[i]), binwidth=0.125, alpha=0.5, ax=ax_dict[method][Q])
            # ax_dict[method][Q].hist(residuals[Q][method][:,0], label=legend_dict[dataset], histtype="step", color=cmap(colors[i]), bins=50)
            ax_dict[method][Q].hist(res/true, label=legend_dict[dataset], histtype="step", color=cmap(colors[i]), bins=50, cumulative=True, density=True, linewidth=1.2)
        
            # sns.kdeplot(residuals[Q][method][:,1], label="True",fill=False, alpha=0.5,color="k", ax=ax_dict[method][Q])
            # sns.kdeplot(residuals[Q][method][:,0], label=legend_dict[dataset],fill=False, alpha=0.5,color=cmap(colors[i]), ax=ax_dict[method][Q])
           
            ax_dict[method][Q].set_xlabel("")
            if k%3 == 0:
                ax_dict[method][Q].set_ylabel(f"{label_dict[method]}")
            else:
                ax_dict[method][Q].set_ylabel("")
        


lines, labels = ax[2,2].get_legend_handles_labels()
pretty_legends = [legend_dict[dataset] for dataset in all_residuals.keys()]
datasets = lines

legend1 = pyplot.legend(datasets, pretty_legends, loc="upper center", bbox_to_anchor=(1.63, 2.6), fancybox=True, fontsize=FONTSIZE, title = "Relative residuals for predictions \nmade by the datasets:", title_fontsize=FONTSIZE)
fig.add_artist(legend1)

fig.text(0.05, 0.23, "Proportions of Relative Residuals by:", ha="center", rotation="vertical", fontsize=FONTSIZE)
fig.text(0.5, 0.02, "Relative Residuals", ha="center", fontsize=FONTSIZE)
plt.xlim(-1.7,2.3)
plt.savefig(f"./../figures/{outfile_name}", bbox_inches="tight")

"""