import matplotlib.pyplot as plt
from matplotlib import pyplot
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



# Name of dataframe file containing true targets
test_df_name = "test_imputed_cleaned_dataframe_Q1_Q3_{Q}.csv"
train_df_name = "train_imputed_cleaned_dataframe_Q1_Q3_{Q}.csv"

# Target preprocessing
target_preprocessing = "mean"  

# Target column
target_col = f"{target_preprocessing}_scl"

# Name of the file to store the predictions
outfile_name = "dist_preds_pca_all_items_Q4_Q5_Q6.pdf"


###################### Load Predictions ################################

predictions = {}

preds_agg = {"linear_regression": ["preds_true_19.04.2022-190154.npy", "preds_true_19.04.2022-191927.npy", "preds_true_19.04.2022-191942.npy"],
               "elastic_net": "preds_true_21.04.2022-084203.npy",
                "neural_net": "preds_true_18.04.2022-130831.npy",
               "boosting":  ["preds_true_21.04.2022-084715.npy", "preds_true_21.04.2022-084856.npy", "preds_true_21.04.2022-131239.npy"]
              }

preds_item = {"linear_regression": ["preds_true_22.04.2022-085119.npy", "preds_true_22.04.2022-092056.npy", "preds_true_22.04.2022-092112.npy"],
               "elastic_net": "preds_true_22.04.2022-084648.npy",
                "neural_net": "preds_true_21.04.2022-190115.npy",
               "boosting":  "preds_true_22.04.2022-085156.npy"
              }

preds_wide = {"linear_regression": ["preds_true_12.04.2022-114130.npy", "preds_true_12.04.2022-122143.npy", "preds_true_29.04.2022-183358.npy"],
               "elastic_net": "preds_true_19.04.2022-175857.npy",#"preds_true_12.04.2022-114232.npy",
                "neural_net": "preds_true_12.04.2022-120420.npy",
               "boosting":  "preds_true_16.04.2022-125845.npy"
              }

preds_pca = {"linear_regression": ["preds_true_07.05.2022-113253.npy", "preds_true_07.05.2022-113308.npy", "preds_true_07.05.2022-113322.npy"],
               "elastic_net": "preds_true_22.04.2022-095832.npy",
                "neural_net": "preds_true_25.04.2022-185205.npy",
               "boosting":  "preds_true_27.04.2022-183454.npy"
              }

preds_corr = {"linear_regression": ["preds_true_29.04.2022-173755-high_corr.npy", "preds_true_29.04.2022-164434-high_corr.npy", "preds_true_29.04.2022-164354-high_corr.npy"],
               "elastic_net": "preds_true_29.04.2022-141045-high_corr.npy",
                "neural_net": "preds_true_28.04.2022-181748-high_corr.npy",
               "boosting":  "preds_true_28.04.2022-160225-high_corr.npy"
              }

all_preds = {"format": preds_wide, "pca": preds_pca} #"aggregated": preds_agg, "item": preds_item, }
# all_preds = {"corr": preds_corr}

methods =  list(preds_agg.keys())

for dataset, preds in all_preds.items():
    if dataset == "corr":
        dataset = "format"
    predictions[dataset] = {}
    
    for i, Q in enumerate(Qs):

        predictions[dataset][Q] = {}

        for method in methods:
            
            if isinstance(preds[method], list) is True:
                path = os.path.join(f"../experiments/Q1_Q3_{Q}/wide_{dataset}_analysis/{method}/results/{preds[method][i]}")
            else: 
                path = os.path.join(f"../experiments/Q1_Q3_{Q}/wide_{dataset}_analysis/{method}/results/{preds[method]}")

            predictions[dataset][Q][method] = np.load(path)

##############################################################################


fig, ax = plt.subplots(nrows=4, ncols=3, sharey=True, sharex = True, figsize=(12,9))
 
cmap = plt.cm.get_cmap("jet")

ax_dict = {
    "linear_regression": {"Q4": ax[0,0], "Q5": ax[0,1], "Q6": ax[0,2]},
    "elastic_net": {"Q4": ax[1,0], "Q5": ax[1,1], "Q6": ax[1,2]},
    "neural_net": {"Q4": ax[2,0], "Q5": ax[2,1], "Q6": ax[2,2]},
    "boosting": {"Q4": ax[3,0], "Q5": ax[3,1], "Q6": ax[3,2]}
}



colors = [0.15, 0.55, 0.75, 0.35]

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


for i, (dataset, preds) in enumerate(predictions.items()):

    for j, method in enumerate(methods):

        for k, Q in enumerate(Qs):

            if method == "linear_regression":
                ax_dict[method][Q].set_title(f"{Q}", fontsize=FONTSIZE) 
                
            
            sns.histplot(preds[Q][method][:,1], label="True Targets", fill=True, color="k",alpha=0.05,binwidth=0.125, ax=ax_dict[method][Q])
            sns.histplot(preds[Q][method][:,0], label=legend_dict[dataset], fill=True, color=cmap(colors[i]), binwidth=0.125, alpha=0.5, ax=ax_dict[method][Q])
            
            # sns.kdeplot(preds[Q][method][:,1], label="True",fill=False, alpha=0.5,color="k", ax=ax_dict[method][Q])
            # sns.kdeplot(preds[Q][method][:,0], label=legend_dict[dataset],fill=False, alpha=0.5,color=cmap(colors[i]), ax=ax_dict[method][Q])
           
            ax_dict[method][Q].set_xlabel("")
            if k%3 == 0:
                ax_dict[method][Q].set_ylabel(f"{label_dict[method]}")
            else:
                ax_dict[method][Q].set_ylabel("")
        


lines, labels = ax[2,2].get_legend_handles_labels()
pretty_legends = [legend_dict[dataset] for dataset in all_preds.keys()]
datasets = lines[1::2]

legend1 = pyplot.legend(datasets, pretty_legends, loc="upper center", bbox_to_anchor=(1.63, 2.6), fancybox=True, fontsize=FONTSIZE, title = "Predictions made by different \nmethods trained on the datasets:", title_fontsize=FONTSIZE)
legend2 = pyplot.legend([lines[0]], ["True Targets                         "], loc="upper center", bbox_to_anchor=(1.63, 1.88), fancybox=True, fontsize=FONTSIZE)
fig.add_artist(legend1)
fig.add_artist(legend2)
plt.xticks(np.linspace(1,4,7), np.linspace(1,4,7))

fig.text(0.05, 0.3, "Distribution of Predictions by:", ha="center", rotation="vertical", fontsize=FONTSIZE)
fig.text(0.5, 0.02, r"Mean SCL Scores In $\mathcal{D}_{test}$", ha="center", fontsize=FONTSIZE)
plt.xlim(0.7,4)
plt.savefig(f"./../figures/{outfile_name}", bbox_inches="tight")