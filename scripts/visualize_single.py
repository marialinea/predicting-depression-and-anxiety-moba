import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

import os
import sys
import pdb



def wide_format_coefs(path, method):
    
    if method != "boosting":

        coef_df = pd.read_csv(path)

        if "Unnamed: 0" in coef_df.columns:
            coef_df.drop(columns=["Unnamed: 0"], inplace=True)

    else:

        coef_df = pd.read_csv(path)
        coef_df.rename(columns={"Unnamed: 0": "features", "score": "coef"}, inplace=True)

   
    return coef_df


def average_topic_coefs(df_dict, short_method):

    path = "../data/all_variables_reversed.json"
    FONTSIZE=12

    rename_dict={
        "scl": "SCL",
        "Work_Q6": "Work Situation",
        "StrainsWork": "Strains at Work",
        "SocialSupport": "Social Support",
        "SWLS": "Satisfaction With Life",
        "RSS": "Relationship \nSatisfaction",
        "RSES": "Self Esteem",
        "LTHofMD": "Lifetime History \nof Major Depression",
        "GSE": "General Self Efficacy",
        "Emotion": "Experience of \nSpecific Emotions",
        "EatingDisorder": "Feelings About Weight",
        "Drugs": "Drug Use",
        "CivilStatus": "Civil Status",
        "SickLeave_Q4": "Sick Leave",
        "Assault_Q6": "Experience of \nAssault",
        "Alcohol": "Alcohol Consumption",
        "Abuse": "Experience of Abuse",
        "AdverseLifeEvents": "Adverse Life Events",
        "Birth": "Birth Experience",
        "ChildDevelopment_Q4": "Child Development",
        "ChildDevelopment_Q5": "Child Development",
        "ChildDevelopment_Q6": "Child Development",
        "ChildCare": "Child Care",
        "ChildMood": "Child Mood",
        "ChildTemperament": "Child Temperament",
        "ChildBehaviour": "Child Behaviour",
        "ChildManner": "Child Manner",
        "Smoking_Q4": "Smoking",
        "Smoking_Q5": "Smoking",
        "Smoking_Q6": "Smoking",
        "Finance": "Finance",
        "ChildLengthWeight": "Child's Length&Weight",
        "Communication": "Child's Communication",
        "Daycare": "Daycare",
        "LivingWithFather": "Living With Father",
        "LivingSituation": "Living Situation",
        "LivingEnvironment": "Living Environment",
        "TimeOutside": "Time spent Outside",
        "PregnantNow": "Pregnant Now",
        "ParentalLeave": "Parental Leave",
        "WHOQOL": "WHOQOL",
        "WalkUnaided": "Walk Unaided",
        "SocialSkills": "Child's Social \nSkills",
        "SocialCommunication": "Social Communication",
        "AdultADHD": "Adult ADHD",
        "PLOC": "Parental Control"
    }

    with open(path, "r") as f:
        variables = json.load(f)

    cmap = plt.cm.get_cmap('jet')
 

    l = {}
    for Q, df in df_dict.items():
        
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

        sorted_idx = np.argsort(mean_values)
        mean_values = mean_values[sorted_idx]
        topics = topics[sorted_idx]
        

        d = {}
        for mean, topic in zip(mean_values, topics):
            d[topic] = mean
        l[Q] = d


    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,14))
    FONTSIZE=16
    ax_dict = {"Q4": ax[0],
                "Q5": ax[1],
                "Q6": ax[2]}

    color_dict = {"Q4": cmap(0.15),
                "Q5": cmap(0.55),
                "Q6": cmap(0.75)}

   
    for Q, features in l.items():

        for i, feature in enumerate(features):
            ax_dict[Q].scatter(l[Q][feature], y=i, color=color_dict[Q], s=60)

            if short_method != "xgb":
                ax_dict[Q].hlines(y=i, xmin=-0.015, xmax=0.015, color="k", alpha=0.3, linewidth=0.7)
            else:
                ax_dict[Q].hlines(y=i, xmin=0, xmax=4, color="k", alpha=0.3, linewidth=0.7)


        ax_dict[Q].set_title(f"{Q}", fontsize=FONTSIZE)
        ax_dict[Q].set_yticks([i for i in range(len(features))])
        ax_dict[Q].set_yticklabels(features.keys(), fontdict={"fontsize":FONTSIZE})
        ax_dict[Q].tick_params(axis="x", labelsize=FONTSIZE)

        if Q == "Q5":
            if short_method != "xgb":
                ax_dict[Q].set_xlabel("Mean Regression Coefficients", fontsize=FONTSIZE)
            else:
                ax_dict[Q].set_xlabel("Mean Gain Scores", fontsize=FONTSIZE)
    
    plt.tight_layout()
    plt.savefig(f"../figures/{short_method}_single_feature_importance.pdf")


def ranked_coefs(df_dicts, methods, short_methods):

    path = "../data/all_variables_reversed.json"
 

    rename_dict={
        "scl": "SCL",
        "Work_Q6": "Work Situation",
        "StrainsWork": "Strains at Work",
        "SocialSupport": "Social Support",
        "SWLS": "Satisfaction With Life",
        "RSS": "Relationship \nSatisfaction",
        "RSES": "Self Esteem",
        "LTHofMD": "Lifetime History \nof Major Depression",
        "GSE": "General Self Efficacy",
        "Emotion": "Experience of \nSpecific Emotions",
        "EatingDisorder": "Feelings About Weight",
        "Drugs": "Drug Use",
        "CivilStatus": "Civil Status",
        "SickLeave_Q4": "Sick Leave",
        "Assault_Q6": "Experience of \nAssault",
        "Alcohol": "Alcohol Consumption",
        "Abuse": "Experience of Abuse",
        "AdverseLifeEvents": "Adverse Life Events",
        "Birth": "Birth Experience",
        "ChildDevelopment_Q4": "Child Development",
        "ChildDevelopment_Q5": "Child Development",
        "ChildDevelopment_Q6": "Child Development",
        "ChildCare": "Child Care",
        "ChildMood": "Child Mood",
        "ChildTemperament": "Child Temperament",
        "ChildBehaviour": "Child Behaviour",
        "ChildManner": "Child Manner",
        "Smoking_Q4": "Smoking",
        "Smoking_Q5": "Smoking",
        "Smoking_Q6": "Smoking",
        "Finance": "Finance",
        "ChildLengthWeight": "Child's Length&Weight",
        "Communication": "Child's Communication",
        "Daycare": "Daycare",
        "LivingWithFather": "Living With Father",
        "LivingSituation": "Living Situation",
        "LivingEnvironment": "Living Environment",
        "TimeOutside": "Time spent Outside",
        "PregnantNow": "Pregnant Now",
        "ParentalLeave": "Parental Leave",
        "WHOQOL": "WHOQOL",
        "WalkUnaided": "Walk Unaided",
        "SocialSkills": "Child's Social \nSkills",
        "SocialCommunication": "Social Communication",
        "AdultADHD": "Adult ADHD",
        "PLOC": "Parental Control"
    }

    with open(path, "r") as f:
        variables = json.load(f)

    cmap = plt.cm.get_cmap('jet')
 

    all_coefs = {}
    for method, df_dict in df_dicts.items():
        l = {}

        for Q, df in df_dict.items():
            
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
            mean_values = np.abs(mean_values)
            topics = np.array(topics)

            sorted_idx = np.argsort(mean_values)[::-1]
            mean_values = mean_values[sorted_idx]
            topics = topics[sorted_idx]
            

            d = {}
            for mean, topic in zip(mean_values, topics):
                d[topic] = mean
            l[Q] = d

        all_coefs[method] = l

    

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,18))
    FONTSIZE=18
    ax_dict = {"Q4": ax[0],
                "Q5": ax[1],
                "Q6": ax[2]}

    color_dict = {"linear_regression": cmap(0.15),
                "elastic_net": cmap(0.55),
                "boosting": cmap(0.75)}
    marker_dict = {"linear_regression": "o",
                "elastic_net": "^",
                "boosting": "s"}

    label_dict = {"linear_regression": "Linear Regression",
               "elastic_net": "Elastic Net",
                "boosting": "XGBoost"
    }

    feature_map = {}
    for Q, features in all_coefs["linear_regression"].items():
        keys = features.keys()

        key_map = {}
        for i, key in enumerate(keys):
            key_map[key] = i

        feature_map[Q] = key_map

    for method, l in all_coefs.items():
       
        for Q, features in l.items():
            values = np.fromiter(features.values(), dtype="f16")
            topics = np.fromiter(features.keys(), dtype="S128")


            for i, feature in enumerate(features):

                ax_dict[Q].scatter(x=i, y=feature_map[Q][feature], color=color_dict[method], s=80, label=f"{method}", alpha=0.8, marker=marker_dict[method])
                # pdb.set_trace()
                # jitter(x=int(i), y=feature_map[Q][feature], color=color_dict[method], s=80, label=f"{method}", ax=ax_dict[Q], alpha=0.8)
                ax_dict[Q].hlines(y=i, xmin=0, xmax=len(values), color="k", alpha=0.3, linewidth=0.7)
            

         
            ax_dict[Q].set_title(f"{Q}", fontsize=FONTSIZE)
            ax_dict[Q].set_yticks([i for i in range(len(features))])
            ax_dict[Q].set_yticklabels(feature_map[Q].keys(), fontdict={"fontsize":FONTSIZE})
            ax_dict[Q].set_xticks([i for i in range(0,len(features), 4)])
            ax_dict[Q].set_xticklabels([i for i in range(1,len(features)+1,4)], fontdict={"fontsize":FONTSIZE})
            ax_dict[Q].tick_params(axis="x", labelsize=FONTSIZE)

            if Q == "Q5":
                ax_dict[Q].set_xlabel("Rank", fontsize=FONTSIZE)

        

    lines, labels = ax_dict[Q].get_legend_handles_labels()
    labels = np.array(labels)

    unique_id1 = np.where(labels == "linear_regression")[0][0]
    unique_id2 = np.where(labels == "elastic_net")[0][0]
    unique_id3 = np.where(labels == "boosting")[0][0]

    lines = [lines[unique_id1], lines[unique_id2], lines[unique_id3]]
    pretty_labels = [label_dict[label] for label in methods]
    
    plt.tight_layout(pad=5)
    
    fig.legend(lines, pretty_labels, loc="lower center", bbox_to_anchor=(0.55, 0), ncol=3, fancybox=True, fontsize=FONTSIZE)
        
    plt.savefig(f"../figures/ranked_coefs_across_time.pdf")

def rand_jitter(arr):
    stdev = 0.05
    return arr + np.random.randn(1) * stdev

def jitter(x, y, s=20, color=None, marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, ax=None, **kwargs):
    return ax.scatter(x, rand_jitter(y), s=s, color=color, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)
    
def plot_dists(id_dict, method, short_method):

    cmap = plt.cm.get_cmap('jet')
 
    path_dict = {
    "linear_regression": "../experiments/{Q}/single_analysis/linear_regression/results",
    "elastic_net": "../experiments/{Q}/single_analysis/elastic_net/results",
    "boosting": "../experiments/{Q}/single_analysis/boosting/results"
    }

    l = {}
    for Q, ids in id_dict.items():
        
        unique_id = ids[method]
        path = path_dict[method].format(Q=Q)

        preds_true = np.load(f"{path}/preds_true_{unique_id}.npy")
        l[Q] = preds_true



    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,14))
    FONTSIZE=16
    ax_dict = {"Q4": ax[0],
                "Q5": ax[1],
                "Q6": ax[2]}

    color_dict = {"Q4": cmap(0.15),
                "Q5": cmap(0.55),
                "Q6": cmap(0.75)}

   
    for Q, features in l.items():
        pdb.set_trace()
        ax_dict[Q].hist(l[Q][:,0], color=color_dict[Q])
        ax_dict[Q].hist(l[Q][:,1], color="k", fill=False)



        ax_dict[Q].set_title(f"{Q}", fontsize=FONTSIZE)
        # ax_dict[Q].set_yticks([i for i in range(len(features))])
        # ax_dict[Q].set_yticklabels(features.keys(), fontdict={"fontsize":FONTSIZE})
        # ax_dict[Q].tick_params(axis="x", labelsize=FONTSIZE)

        # ax_dict[Q].set_xlabel("Mean Regression Coefficients", fontsize=FONTSIZE)
           
    
    plt.tight_layout()
    plt.savefig(f"../figures/{short_method}_single_dist.pdf")
    



if __name__=="__main__":

    id_dict = {
        "Q4": {
            "linear_regression": "19.04.2022-164632",
            "elastic_net": "19.04.2022-200007",
            "boosting": "01.05.2022-172749"
        },
        "Q5": {
            "linear_regression": "19.04.2022-165331",
            "elastic_net": "19.04.2022-200025",
            "boosting": "04.05.2022-191952"
        },
        "Q6": {
            "linear_regression": "19.04.2022-165427",
            "elastic_net": "19.04.2022-200044",
            "boosting": "05.05.2022-110238"
        }
    }

    path_dict = {
        "linear_regression": "../experiments/{Q}/single_analysis/linear_regression/coefficients/lr_coefficients_{unique_id}.csv",
        "elastic_net": "../experiments/{Q}/single_analysis/elastic_net/coefficients/en_coefficients_{unique_id}.csv",
        "boosting": "../experiments/{Q}/single_analysis/boosting/feature_importance/{unique_id}/feature_importance.csv"
    }


    # method = "linear_regression"
    # short_method = "lr"

    # method = "elastic_net"
    # short_method = "en"

    method = "boosting"
    short_method = "xgb"


    df_dict = {}

    # for Q, method_dict in id_dict.items():
        
    #     unique_id = method_dict[method]
    #     path = path_dict[method].format(Q=Q, unique_id=unique_id)

    #     df_dict[Q] = wide_format_coefs(path, method)

    # average_topic_coefs(df_dict, short_method)
    # plot_dists(id_dict, method, short_method)

    methods = ["linear_regression", "elastic_net", "boosting"]
    short_methods = ["lr", "en", "xgb"]

    all_coefs = {}
    for method in methods:

        df_dict = {}

        for Q, method_dict in id_dict.items():
            
            unique_id = method_dict[method]
            path = path_dict[method].format(Q=Q, unique_id=unique_id)

            df_dict[Q] = wide_format_coefs(path, method)
        
        all_coefs[method] = df_dict

    ranked_coefs(all_coefs, methods, short_methods)