import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import pdb
import json

from visualize_coeffs import make_line_plot

            
def variables_from_index(index_vec):

    path = "../data/all_variables_reversed.json"

    with open(path, "r") as f:
        variables = json.load(f)

    feature_path = "../experiments/Q1_Q3_Q4/wide_pca_analysis/linear_regression/coefficients/features.txt"
    features = []
            
    with open(feature_path, "r") as infile:
        for line in infile.readlines():
            line = line.split("\n")[0]
            features.append(line)

    features = [features[i] for i in index_vec]
    for i, f in zip(index_vec, features):
        try:
            print(f"{i} - {f}: {variables[f]}")
        except KeyError:
            print(f"{i} - {f}")

def combine_pca():
    wide_analysis = ["Q1_Q3_Q4", "Q1_Q3_Q5", "Q1_Q3_Q6"]

    # analysis = "wide_format"
    analysis = "wide_pca"

    experiment_path = "./../experiments"
    paths = {}


    for i in range(len(wide_analysis)):
        paths[wide_analysis[i]] = os.path.join(experiment_path, wide_analysis[i], f"{analysis}_analysis")



    configs = {
        "wide_format": ["18.03.2022-194109", "18.03.2022-194059", "18.03.2022-194148"],
        "wide_pca": ["21.03.2022-012341", "21.03.2022-012341", "21.03.2022-012341"]
    }
    

    method = "boosting"

    feature_importance = []

    top_n = 20

    for i in range(len(wide_analysis)):

        config = configs[analysis][i]
        imp_path = os.path.join(paths[wide_analysis[i]], method, "feature_importance", f"{config}", "feature_importance.csv")
        
        df = pd.read_csv(imp_path)

        if "Unnamed: 0" in df.columns:
                if analysis == "wide_format":
                    df = df.rename(columns={"Unnamed: 0": "features"})
                    col_name = "features"
                elif analysis == "wide_pca":
                    df = df.rename(columns={"Unnamed: 0": "principal_comps"})
                    col_name = "principal_comps"

        top_imp = df[[f"{col_name}", "score"]][1:top_n+1]
        
        if analysis == "wide_format":

            print("-"*30)
            variables_from_index(top_imp["features"])

        feature_importance.append(top_imp)
        # pdb.set_trace()


    if analysis == "wide_pca":
        # Get features
        feature_path = "../experiments/Q1_Q3_Q4/wide_pca_analysis/linear_regression/coefficients/features.txt"
        features = []
        
        with open(feature_path, "r") as infile:
            for line in infile.readlines():
                line = line.split("\n")[0]
                features.append(line)


    figpath=f"../figures/xgb_{analysis}_20x20_pca_Q4_Q5_Q6.pdf"

    make_line_plot(feature_importance, wide_analysis, top_n, features, figpath, method)

def combine_gain():

    fig, axs = plt.subplots(1,3, figsize=(26,10), sharey = True)
    SMALLSIZE = 16
    BIGSIZE=30
    cmap = plt.cm.get_cmap('mako')
    color = cmap(0.6)

    for i in range(len(wide_analysis)):
        Q = wide_analysis[i].split("_")[-1]

        df = feature_importance[i]
        sns.barplot(x=df["features"], y=df["score"], order=df["features"], ax=axs[i], color=color)
        axs[i].set_title(f"{Q}", fontdict={"fontsize":BIGSIZE})

        axs[i].set_ylabel("")
        axs[i].set_xlabel("")

        if i==0:
            axs[i].set_ylabel("Gain", fontsize=BIGSIZE)
        if i == 1:
            if analysis == "wide_format":
                axs[i].set_xlabel("Features", fontsize=BIGSIZE)
            elif analysis == "wide_pca":
                axs[i].set_xlabel("Principal Components", fontsize=BIGSIZE)

        
        axs[i].tick_params(axis="x", labelsize=SMALLSIZE)
        axs[i].tick_params(axis="y", labelsize=SMALLSIZE)

        plt.setp(axs[i].get_xticklabels(), rotation=40, horizontalalignment='right')


    fig.tight_layout()
    plt.savefig(f"./../figures/boosting_gain_{analysis}_Q4_Q5_Q6.pdf")


        # FONTSIZE=14

        # x = data_df.nlargest(num_features, columns="score")["features"]
        # y = data_df.nlargest(num_features, columns="score")["score"]
        
        # fig, ax = plt.subplots()

        # sns.barplot(x=x, y=y, color="aquamarine")
        # plt.ylabel("Gain", fontsize=FONTSIZE)
        # plt.xlabel("Features", fontsize=FONTSIZE)
        # plt.xticks(fontsize=FONTSIZE),
        # plt.yticks(fontsize=FONTSIZE)
        # plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right')

        # plt.savefig(fig_path, bbox_inches="tight")
        # plt.close()