import pandas as pd
import numpy as np
import os
import json
import pdb
import sys
import re

from scipy.stats import f_oneway

from utils.print import print_blue



analysis = sys.argv[1]

if analysis == "agg":
    analysis = "aggregated"
elif analysis == "item":
    analysis = "item_level"
elif analysis == "df":
    analysis = "complete_df"
elif analysis == "wide":
    analysis = "wide_format"

print("")
print_blue(f"Printing metrics for {analysis} analysis \n")


long_experiments = ["aggregated", "item_level", "complete_df", "pca"]

if analysis in long_experiments: 
    Q_names = ["Q1", "Q3", "Q4", "Q5", "Q6"]

elif "wide" in analysis:
    Q_names = ["Q1", "Q3", "Q5"]


methods = {"linear_regression": "lr",
        "elastic_net": "en",
        "neural_net": "nn", 
        "boosting": "boosting"
        }

result_path = "./experiments/" + "_".join(Q_names) + "/{analysis}_analysis/{method}/results/"

config_path = "./experiments/" + "_".join(Q_names) + "/{analysis}_analysis/{method}/configs/"


results = {}
for method in methods.keys():
    results[method] = {}
    
    files = os.listdir(result_path.format(analysis=analysis, method=method))
    count = 0
    for fn in files:
        
        metric_file = methods[method] + "_metrics_"

        if metric_file in fn:
            
            count +=1

            
            file_dict = {}
            

            with open(result_path.format(analysis=analysis, method=method) + fn, "r") as f:
                metrics = json.load(f)
            
            if analysis in long_experiments:

                for Q in Q_names:

                    rmse =  metrics[Q]["rmse"]
                    file_dict[Q] = rmse

            elif "wide" in analysis:
                rmse =  metrics["rmse"]
                file_dict["rmse"] = rmse
         

            # Get config name
            if os.path.isdir(config_path.format(analysis=analysis, method=method)) is True:

                words = re.split("\.|_|-", fn)
                date_time = [s for s in words if s.isdigit()]

                config_name = f"config_{date_time[0]}.{date_time[1]}.{date_time[2]}-{date_time[-1]}"
            else:
                config_name = ""

            file_dict["config"] = config_name

            results[method][f"file{count}"] = file_dict
        

print_list = [] 
for key, values in results.items():
    for _, d in values.items():
        l = []
        l.append(key)
        for __, v in d.items():
            l.append(v)
        print_list.append(l)    
        

# pdb.set_trace() 

if analysis in long_experiments:

    print("{:<20s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s}  {:^8s} {:^28s}".format("Method", *Q_names, "Mean", "Config"))
    print("--"*50)

    results = {}
    results["Method"] = []
    results["Q1"] = []
    results["Q3"] = []
    results["Q4"] = []
    results["Q5"] = []
    results["Q6"] = []
    results["Config"] = []



    for i in range(len(print_list)):
        results["Method"].append(print_list[i][0])
        results["Q1"].append(print_list[i][1])
        results["Q3"].append(print_list[i][2])
        results["Q4"].append(print_list[i][3])
        results["Q5"].append(print_list[i][4])
        results["Q6"].append(print_list[i][5])
        results["Config"].append(print_list[i][6])


    df = pd.DataFrame(results)


    for Q in Q_names:
        rmse_score = results[Q].copy()
        min_score = np.argmin(results[Q])
        max_score = np.argmax(results[Q])

        results[Q] = []
        for i, score in enumerate(rmse_score):
            if i == min_score:
                s = f"\x1b[92m{score:.4f}\033[0m"
                results[Q].append(s)
            elif i == max_score:
                s = f"\033[91m{score:.4f}\033[0m"
                results[Q].append(s)
            else:
                results[Q].append(f"{score:.4f}")
        

    columns = df.columns
    means = df[columns[1:-1]].mean(axis=1)

    color_mean = []
    min_score = np.argmin(means)
    max_score = np.argmax(means)
    for i, score in enumerate(means):
            if i == min_score:
                s = f"\x1b[92m{score:.4f}\033[0m"
                color_mean.append(s)
            elif i == max_score:
                s = f"\033[91m{score:.4f}\033[0m"
                color_mean.append(s)
            else:
                color_mean.append(f"{score:.4f}")



    for a, b, c, d, e, f, g, h in zip(results["Method"], results["Q1"], results["Q3"], results["Q4"], results["Q5"], results["Q6"], results["Config"], color_mean):
        print("{:<20s} | {:^1s} | {:^1s} | {:^1s} | {:^1s} | {:^1s} | {:^1s} |  {:>20s}".format(a,b,c,d,e,f,h,g))

elif "wide" in analysis:
    print("{:<20s} {:^8s} {:^28s}".format("Method", "rmse", "Config"))
    print("--"*30)

    results = {}
    results["Method"] = []
    results["rmse"] = []
    results["Config"] = []


    
    for i in range(len(print_list)):
        results["Method"].append(print_list[i][0])
        results["rmse"].append(print_list[i][1])
        results["Config"].append(print_list[i][2])


    df = pd.DataFrame(results)


 
    rmse_score = results["rmse"].copy()
    min_score = np.argmin(rmse_score)
    max_score = np.argmax(rmse_score)
    rmse = []

    for i, score in enumerate(rmse_score):
        if i == min_score:
            s = f"\x1b[92m{score:.4f}\033[0m"
            rmse.append(s)
        elif i == max_score:
            s = f"\033[91m{score:.4f}\033[0m"
            rmse.append(s)
        else:
            rmse.append(f"{score:.4f}")
        




    for a, b, c in zip(results["Method"], rmse, results["Config"]):
        print("{:<20s} | {:^1s} | {:>20s}".format(a,b,c))





# # One way ANOVA

# print_blue("\nPerforming one way ANOVA \n")

# if analysis == "aggregated":



#     best_rows = [0, 2, 12]

#     best_results = df[columns[1:-1]].loc[best_rows].to_numpy()

#     statistic, pvalue = f_oneway(best_results[0], best_results[1], best_results[2])

# if analysis == "item_level":

#     best_rows = [0, 7, 9]

#     best_results = df[columns[1:-1]].loc[best_rows].to_numpy()

#     statistic, pvalue = f_oneway(best_results[0], best_results[1], best_results[2])


# if pvalue > 0.05:
#     print(f"F test statistic: {statistic:.3f} \np-value: \033[91m{pvalue:.3f}\033[0m \n")
# else:
#     print(f"F test statistic: {statistic:.3f} \np-value: \x1b[92m{pvalue:.3f}\033[0m \n")
  
