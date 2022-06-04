import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

import os
import sys
import pdb




def extract_top_N_coefs(coefs, features, N):

    sorted_idx = sort_coefs(coefs)


    sorted_coefs = coefs[sorted_idx]
    sorted_features = features[sorted_idx]

    if "Intercept" in list(sorted_features.values):
        idx = np.where(sorted_features.values == "Intercept")[0]
        sorted_features.drop(index=idx, inplace=True)
        sorted_coefs.drop(index=idx, inplace=True)

    top_coefs = sorted_coefs[:N]
    top_features = sorted_features[:N]

    top_df = pd.DataFrame([top_features, top_coefs]).T

    return top_df

    

def sort_coefs(coefs):

    abs_coefs = np.abs(coefs)

    # Largest value in coefs correspnd with the first index in sorted_idxs array
    sorted_idxs = np.argsort(abs_coefs)[::-1]

    return sorted_idxs


def wide_format_coefs(path, method, N=None):
    
    if method == "linear_regression":

        coef_df = pd.read_csv(path)

        if "Unnamed: 0" in coef_df.columns:
            coef_df.drop(columns=["Unnamed: 0"], inplace=True)

    elif method == "elastic_net":

        with open(path, "r") as f:
            coef_json = json.load(f)

        coef_dict = {}
        coef_dict["features"] = []
        coef_dict["coef"] = []
        for key, value in coef_json.items():
            coef_dict["features"].append(key)
            coef_dict["coef"].append(value)

       
        coef_df = pd.DataFrame.from_dict(coef_dict)

    elif method == "boosting":

        coef_df = pd.read_csv(path)
        coef_df.rename(columns={"Unnamed: 0": "features", "score": "coef"}, inplace=True)

    coefs = coef_df["coef"]
    features = coef_df["features"]

    if N is not None:
        top_df = extract_top_N_coefs(coefs, features, N)

        return top_df

    else: 
        return coef_df

def wide_pca_coefs(path, N):

    reg_coefs = pd.read_csv(path)
  
    if "Unnamed: 0" in reg_coefs.columns:
        reg_coefs.drop(columns=["Unnamed: 0"], inplace=True)

    principal_comp = reg_coefs["principal_comps"]
    coefs = reg_coefs["coef"]

    top_pcs = extract_top_N_coefs(coefs, principal_comp, N)

    top_pcs["principal_comps"] = top_pcs["principal_comps"].astype("int")
    
    return top_pcs

def make_line_plot(df_list, Qs, N,features, figpath, method):
    """
    Makes pie chart over topics of the top 20 coefficients in the top 20 principal components
    for each time step.
    """

    # Load feature topic dict
    path = "../data/all_variables_reversed.json"

    with open(path, "r") as f:
        variables = json.load(f)
    
    # Load principal components
    pc_path = "../experiments/Q1_Q3_Q4/wide_pca_analysis/linear_regression/coefficients/lr_principle_componenets_07.05.2022-113253.npy"
  

    pcs = np.load(pc_path, allow_pickle=True)

    rename_dict={
        "scl": "SCL",
        "Work_Q3": "Work Situation Q3",
        "Work": "Work Situation Q1",
        "StrainsWork": "Strains at Work",
        "SocialSupport": "Social Support",
        "SickLeave_Q3": "Sick Leave at Q3",
        "SWLS": "Satisfaction With Life",
        "RSS": "Relationship \nSatisfaction",
        "RSES": "Self Esteem",
        "PreviousPregnancy": "Previous \nPregnancies",
        "LTHofMD": "Lifetime History \nof Major Depression",
        "IncomeHousing": "Income&Housing",
        "Household": "Living Situation",
        "HeightWeight": "Height&Weight",
        "GSE": "General Self Efficacy",
        "Emotion": "Experience of \nSpecific Emotions",
        "Edu": "Education",
        "EatingDisorder": "Feelings About Weight",
        "Drugs": "Drug Use",
        "CivilStatus": "Civil Status",
        "Assault_Q3": "Experience of \nAssault at Q3",
        "Alcohol": "Alcohol Consumption",
        "Abuse": "Experience of Abuse",
        "AdverseLifeEvents": "Adverse Life Events",
        "Language": "Native Language"
    }
    
    count_list = []
    all_topics = []
    for k, (df, Q) in enumerate(zip(df_list, Qs)):
        Q = Q.split("_")[-1]
        
        top_pcs = df["principal_comps"].to_numpy()

        pc = pcs[top_pcs].reshape(len(top_pcs),-1)

        zero_idx = np.where(np.abs(pc) < 1e-16)
        pc[zero_idx] = 0

        abs_pc = np.abs(pc)

        top_indices = np.argsort(abs_pc, axis=-1)
        top_indices = np.flip(top_indices, axis=-1)[:,:N]

        top_features = []
        for i in range(N):
            for j in range(N):
                try:
                    top_features.append(variables[features[top_indices[i,j]]])
                except KeyError:
                    top_features.append(features[top_indices[i,j]])

        top_features = np.array(top_features)

        df = pd.DataFrame(np.unique(top_features, return_counts=True)).T
        df.columns = ["features", "counts"]

        sorted_counts = np.argsort(df["counts"])
        topics = df["features"].iloc[sorted_counts]

        all_topics.extend(list(topics))
    
        df.set_index("features", inplace=True)
        df_dict = df.to_dict()

        count_list.append(df_dict)
        
    # pdb.set_trace()  
    plt.figure(figsize=(6,9))
    FONTSIZE = 14
  

    cmap = plt.cm.get_cmap('jet')
    color = cmap(0.6)

    all_topics = list(set(all_topics))
    # pdb.set_trace()
    topics_sorted = [
        "Alcohol",
        "Work",
        "Assault_Q3",
        "SickLeave_Q3",
        "RSS",
        "Abuse",
        "AdverseLifeEvents",
        "PreviousPregnancy",
        "StrainsWork",
        "EatingDisorder",
        "Emotion",
        "LTHofMD",
        "SWLS",
        "Household",
        "Work_Q3",
        "IncomeHousing",
        "RSES",
        "scl",
        "GSE",
        "SocialSupport",
        "Language",
        "Edu",
        "Drugs",
        "HeightWeight",
        "CivilStatus"
    ]
    topics_sorted.reverse()

    y_values = [i for i in range(len(all_topics))]
    nice_topics = []
    for topic, y in zip(topics_sorted, y_values):
  
        try:    
            plt.scatter(count_list[0]["counts"][topic], y, color=cmap(0.15), alpha=0.9)
        except KeyError:
            plt.scatter(x=0.1, y=y, color=cmap(0.15))
        
        try:
            plt.scatter(count_list[1]["counts"][topic], y, color=cmap(0.55), alpha=.8)
        except KeyError:
            plt.scatter(x=0.1, y=y, color=cmap(0.55))

        try:
            plt.scatter(count_list[2]["counts"][topic], y, color=cmap(0.75),alpha=.5)
        except KeyError:
            plt.scatter(x=0.1, y=y, color=cmap(0.75))
        

        plt.hlines(y=y, xmin=0, xmax=90, color="k", alpha=0.3, linewidth=0.7)

        nice_topics.append(rename_dict[topic])
        
        if y == 0:
            plt.legend(["Q4", "Q5", "Q6"], loc="center right")

    

    plt.yticks([i for i in range(len(all_topics))], nice_topics)

    plt.xlabel("Count", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(figpath)

        # categories = [rename_dict[f] for f in df['features']]
        

        # pdb.set_trace()

        # ax[k].hlines(y=categories, xmin=0, xmax=df["counts"], color=color, alpha=0.7, linewidth=2)
        # ax[k].scatter(y=df.index, x=df["counts"], s=75, color=color, alpha=0.7)
        # plt.setp(ax[k].get_xticklabels(), rotation=0, horizontalalignment='right', fontsize=SMALLSIZE)
        # plt.setp(ax[k].get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=SMALLSIZE)


        # ax[k].set_title(f"{Q}", fontsize=BIGSIZE)
        # ax[k].set_yticks(fontsize=SMALLSIZE)
    # plt.tight_layout(pad=3)
    # fig.text(0.5, 0.01, "Count", ha="center", fontsize=BIGSIZE)
    # fig.text(0.012, 0.45, "Feature Topic", ha="center", rotation="vertical", fontsize=BIGSIZE)
    # plt.savefig(figpath)

        
    
def bubble_triple_plot(df_list, Qs, features, figpath, method, return_zeros=False):

    # Plotting details
    BIGSIZE = 9
    SMALLSIZE = 7

    cmap = plt.cm.get_cmap('mako')
    color = cmap(0.6)

    figsize=(7,3)
    xlabel="Features"
    feature_col="principal_comps"

    # Load principal components
    if method == "linear_regression":
        pc_path = "../experiments/Q1_Q3_Q4/wide_pca_analysis/linear_regression/coefficients/lr_principle_componenets_23.03.2022-123904.npy"
    elif method == "elastic_net":
        raise NotImplementedError

    pcs = np.load(pc_path)

    fig, ax = plt.subplots(figsize=figsize)

    linear_combs = np.empty(shape=(3,410))

    pc_number = []
    zero_coefs = []
    for i, (df, Q) in enumerate(zip(df_list, Qs)):
        Q = Q.split("_")[-1]

        top_pc = df[feature_col].iloc[0]
        pc_number.append(f"{Q}: {top_pc}")

        pc = pcs[top_pc].reshape(1,-1)

        zero_idx = np.where(np.abs(pc) < 1e-16)
        zero_coefs.append(zero_idx[1])
        
        pc[zero_idx] = 0

        linear_combs[i,:] = np.abs(pc)
        
    

    x = [i for i in range(linear_combs.shape[1])]
    y = [i for i in range(linear_combs.shape[0])]

    X, Y = np.meshgrid(x,y)


    plt.scatter(X, Y, s=linear_combs*500, alpha=0.4, color=color)
    plt.yticks([i for i in range(3)], pc_number) 
    fig.text(0.5, 0.01, "Feature Index", ha="center", fontsize=BIGSIZE)
    fig.text(0.015, 0.08, "Top Principal Component per Time Point", ha="center", rotation="vertical", fontsize=BIGSIZE)
    fig.tight_layout(pad=2)
    plt.savefig(figpath)

    if return_zeros is True:

        zero_Q4 = zero_coefs[0]
        zero_Q5 = zero_coefs[1]
        zero_Q6 = zero_coefs[2]

        intersect_Q4Q5 = np.intersect1d(zero_Q4, zero_Q6)
        intersect_all = np.intersect1d(intersect_Q4Q5, zero_Q6)

        return intersect_all
       


def plot_triple(df_list, Qs, figpath, analysis):

    # Plotting details
    BIGSIZE = 14
    SMALLSIZE = 7

    cmap = plt.cm.get_cmap('mako')
    color = cmap(0.6)

    if analysis == "wide_format":
        figsize=(10,4)
        # ylim=(-0.03,0.03)
        ylim=(-0.009,0.009)
        xlabel="Features"
        ylabel="Regression Coefficients"
        feature_col="features"

    elif analysis == "wide_pca":
        figsize=(10,4)

        if df_list[0].columns[0] == "pricipal_comps":

            ylim=(-0.01, 0.01)
            xlabel="Principal Components"
            ylabel="Regression Coefficients"
            feature_col="principal_comps"

        elif df_list[0].columns[0] == "features":
            ylim=(-0.4, 0.4)
            xlabel="Features"
            ylabel="Principal Component Coefficients"
            feature_col="features"


    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)

    for i, (df, Q) in enumerate(zip(df_list, Qs)):
        Q = Q.split("_")[-1]
        
        coefs = np.abs(df["coef"])
        features = df[feature_col].astype("str")
        # pdb.set_trace()

        ax[i].bar(x=features, height=coefs, color=color)
        ax[i].set_title(f"{Q}")
        plt.setp(ax[i].get_xticklabels(), rotation=42, horizontalalignment='right', fontsize=SMALLSIZE)

    fig.text(0.5, 0.01, xlabel, ha="center", fontsize=BIGSIZE)
    fig.text(0.012, 0.25, ylabel, ha="center", rotation="vertical", fontsize=BIGSIZE)
    fig.tight_layout(pad=2)
    plt.ylim(ylim)
    plt.savefig(figpath)


def get_feature_topics(features):

    path = "../data/all_variables_reversed.json"

    with open(path, "r") as f:
        variables = json.load(f)

    for f in features:
        try:
            print(f"{f}: {variables[f]}")
        except KeyError:
            print(f"{f}")


        
        
def find_features_pc(df_list, Qs, features, N, method):
    """
    Function that find the top N features in one principle component
    """

    # Load principal components
    if method == "linear_regression":
        pc_path = "../experiments/Q1_Q3_Q4/wide_pca_analysis/linear_regression/coefficients/lr_principle_componenets_23.03.2022-123904.npy"
    elif method == "elastic_net":
        raise NotImplementedError

    pcs = np.load(pc_path)

    sorted_df_list = []
    for i, (df, Q) in enumerate(zip(df_list, Qs)):
        Q = Q.split("_")[-1]

        top_pc = df["principal_comps"].iloc[0]

        pc = pcs[top_pc]

        sorted_idx = sort_coefs(pc)

        # pdb.set_trace()
        sorted_pc = pc[sorted_idx][:N]
        sorted_features = np.array(features)[sorted_idx][:N]

        df = pd.DataFrame([sorted_features, sorted_pc], index=["features", "coef"]).T
        sorted_df_list.append(df)

    return sorted_df_list

def average_topic_coefs(df_list, Qs, short_method):

    path = "../data/all_variables_reversed.json"
    FONTSIZE=12

    rename_dict={
        "mean_scl_Q3": "Mean SCL at Q3",
        "mean_scl_Q1": "Mean SCL at Q1",
        "scl": "SCL",
        "Work_Q3": "Work Situation Q3",
        "Work": "Work Situation Q1",
        "StrainsWork": "Strains at Work",
        "SocialSupport": "Social Support",
        "SickLeave_Q3": "Sick Leave at Q3",
        "SWLS": "Satisfaction With Life",
        "RSS": "Relationship \nSatisfaction",
        "RSES": "Self Esteem",
        "PreviousPregnancy": "Previous \nPregnancies",
        "LTHofMD": "Lifetime History \nof Major Depression",
        "IncomeHousing": "Income&Housing",
        "Household": "Living Situation",
        "HeightWeight": "Height&Weight",
        "GSE": "General Self Efficacy",
        "Emotion": "Experience of \nSpecific Emotions",
        "Edu": "Education",
        "EatingDisorder": "Feelings About Weight",
        "Drugs": "Drug Use",
        "CivilStatus": "Civil Status",
        "Assault_Q3": "Experience of \nAssault at Q3",
        "Alcohol": "Alcohol Consumption",
        "Abuse": "Experience of Abuse",
        "AdverseLifeEvents": "Adverse Life Events",
        "Language": "Native Language"
    }

    with open(path, "r") as f:
        variables = json.load(f)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,9))
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(25,9))
    ax_dict = {"Q1_Q3_Q4": ax[0],
                "Q1_Q3_Q5": ax[1],
                "Q1_Q3_Q6": ax[2]}

    cmap = plt.cm.get_cmap('jet')
 

    l = {}
    for Q, df in zip(Qs, df_list):
        
        coefs = df["coef"]
        features = df["features"]
        
        # pdb.set_trace()
        mean_contributions = {}
        for coef, feature in zip(coefs, features):
            if feature == "Intercept":
                continue
            try:
                mean_contributions[variables[feature]].append(coef)
            except KeyError:
                mean_contributions[variables[feature]] = []
                mean_contributions[variables[feature]].append(coef)

        
        topics = []
        mean_values = []
     
        for key, values in mean_contributions.items():
            topics.append(rename_dict[key])
            mean_values.append(np.mean(np.array(values)))
       

        mean_values = np.array(mean_values)
        topics = np.array(topics)
        
        if Q == "Q1_Q3_Q4":
            sorted_idx = np.argsort(mean_values)
            sorted_topics = topics[sorted_idx]
        pdb.set_trace()
        d = {}
        for mean, topic in zip(mean_values, topics):
            d[topic] = mean
        l[Q] = d

    #     ax_dict[Q].hlines(y=[i for i in range(len(topics))], xmin=0, xmax=mean_values, linewidth=5)
    #     ax_dict[Q].set_yticks([i for i in range(len(topics))])#, fontsize=FONTSIZE)
    #     ax_dict[Q].set_yticklabels(topics)#, fontsize=FONTSIZE)
    #     ax_dict[Q].grid(linestyle='--', alpha=0.5)    

    # fig.tight_layout()
    # fig.savefig("./../figures/test_bars.pdf")

    y_values = [i for i in range(len(topics))]
    plt.figure(figsize=(6,9))
    
    for topic, y in zip(sorted_topics, y_values):
            
        plt.scatter(l["Q1_Q3_Q4"][topic], y, color=cmap(0.15))#, edgecolor="k", linewidth=0.7,alpha=0.6)
        plt.scatter(l["Q1_Q3_Q5"][topic], y, color=cmap(0.55))
        plt.scatter(l["Q1_Q3_Q6"][topic], y, color=cmap(0.75))
        

        if short_method != "xgb":
            plt.hlines(y=y, xmin=-0.0082, xmax=0.0082, color="k", alpha=0.3, linewidth=0.7)
        else:
            plt.hlines(y=y, xmin=0, xmax=25, color="k", alpha=0.3, linewidth=0.7)
        
        if y == 0:
            plt.legend(["Q4", "Q5", "Q6"], loc="center right")

    plt.yticks([i for i in range(len(topics))], sorted_topics)

    if short_method != "xgb":
        plt.xlabel("Mean Regression Coefficients", fontsize=FONTSIZE)
        figname = f"{short_method}_reg_coefs_dot_plot_Q4_Q5_Q6.pdf"
    else:
        plt.xlabel("Mean Gain Score", fontsize=FONTSIZE)
        figname = f"{short_method}_gain_dot_plot_Q4_Q5_Q6.pdf"

    plt.tight_layout()
    plt.savefig(f"./../figures/{figname}")
    





    

if __name__=="__main__":

    path_dict = {
    "linear_regression": {
        "wide_format": "../experiments/{Qs}/wide_format_analysis/linear_regression/coefficients/lr_coefficients_{unique_id}.csv",
        "wide_pca": "../experiments/{Qs}/wide_pca_analysis/linear_regression/coefficients/lr_coefficients_{unique_id}.csv",
    },
    "elastic_net": {
        "wide_format": "../experiments/{Qs}/wide_format_analysis/elastic_net/coefficients/en_coefficients_{unique_id}.json",
        "wide_pca": "../experiments/{Qs}/wide_pca_analysis/elastic_net/coefficients/en_coefficients_{unique_id}.csv",
    },
    "boosting": {
        "wide_format": "../experiments/{Qs}/wide_format_analysis/boosting/feature_importance/{unique_id}/feature_importance.csv",
        "wide_pca": "../experiments/{Qs}/wide_pca_analysis/boosting/feature_importance/{unique_id}/feature_importance.csv",
    }
}

    id_dict = {
        "linear_regression": {
            "wide_format": { 
                "Q1_Q3_Q4": "12.04.2022-114130",
                "Q1_Q3_Q5": "12.04.2022-122143",
                "Q1_Q3_Q6": "13.04.2022-094536"

            },
            "wide_pca": {
                "Q1_Q3_Q4": "07.05.2022-113253", 
                "Q1_Q3_Q5": "07.05.2022-113308",
                "Q1_Q3_Q6": "07.05.2022-113322"
            }
        },
        "elastic_net": {
            "wide_format": {
                "Q1_Q3_Q4": "19.04.2022-175857",
                "Q1_Q3_Q5": "19.04.2022-175857",
                "Q1_Q3_Q6": "19.04.2022-175857"
            },
            "wide_pca": {
                "Q1_Q3_Q4": "22.04.2022-095832", 
                "Q1_Q3_Q5": "22.04.2022-095832",
                "Q1_Q3_Q6": "22.04.2022-095832"
            }
        },
        "boosting": {
            "wide_format": {
                "Q1_Q3_Q4": "16.04.2022-125845",
                "Q1_Q3_Q5": "16.04.2022-125845",
                "Q1_Q3_Q6": "16.04.2022-125845"
            },
            "wide_pca": {
                "Q1_Q3_Q4": "27.04.2022-183454", 
                "Q1_Q3_Q5": "27.04.2022-183454",
                "Q1_Q3_Q6": "27.04.2022-183454"
            }
        }
    }


    method = "linear_regression"
    method_short = "lr"

    # method = "elastic_net"
    # method_short = "en"

    # method = "boosting"
    # method_short = "xgb"


    analysis = "wide_format"
    # analysis = "wide_pca"

    if analysis == "wide_pca":
        # Get features
        feature_path = "../experiments/Q1_Q3_Q4/wide_pca_analysis/linear_regression/coefficients/features.txt"
        features = []
        
        with open(feature_path, "r") as infile:
            for line in infile.readlines():
                line = line.split("\n")[0]
                features.append(line)

    

    df_list = []

    for Qs, id in id_dict[method][analysis].items():
        
        path = path_dict[method][analysis].format(Qs=Qs, unique_id=id)
        
        if analysis == "wide_format":

            N = None
            df_list.append(wide_format_coefs(path, method, N))

        elif analysis == "wide_pca":
            
            N = 20
            df_list.append(wide_pca_coefs(path, N))

    
    Qs = list(id_dict[method][analysis].keys())
    # pdb.set_trace()
    average_topic_coefs(df_list, Qs, method_short)

    # figpath = f"../figures/{method_short}_coefs_{analysis}_Q4_Q5_Q6.pdf"
    
    # figpath = f"../figures/{method_short}_{analysis}_line_plot_features_Q4_Q5_Q6.pdf"
    # sorted_df_list = find_features_pc(df_list, Qs, features, N, method)
    
    
    # figpath = f"../figures/{method_short}_{analysis}_bubble_Q4_Q5_Q6.pdf"
    # zeros = bubble_triple_plot(df_list, Qs, features, figpath, method, return_zeros=True)
    
    figpath = f"../figures/{method_short}_top_pca_coefs_Q4_Q5_Q6.pdf"
    # plot_triple(sorted_df_list, Qs, figpath, analysis)

    # make_line_plot(df_list, Qs, N, features, figpath, method)
    # plot_triple(df_list, Qs, figpath, analysis)
   
    
    # get_feature_topics([features[i] for i in zeros])
    # path = "../data/all_variables_reversed.json"

    # with open(path, "r") as f:
    #     variables = json.load(f)
    # features = [features[i] for i in zeros]
    # for i, f in zip(zeros, features):
    #     print(f"{i} - {f}: {variables[f]}")

    
    # for i in range(len(df_list)):
    #     df = df_list[i]
    #     features = df["features"]
    #     print("----------------------------")
    #     get_feature_topics(features)