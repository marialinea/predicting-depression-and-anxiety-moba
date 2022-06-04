import matplotlib.pyplot as plt
import seaborn as sns
# from heatmap import corrplot, heatmap
from plot import corrplot
import numpy as np
import pandas as pd
import pdb 


def find_corr(path, Q, top_N):

    df = pd.read_csv(path)
    df.drop(columns=["PREG_ID_2601"], inplace=True)
    df.rename(columns={f"mean_scl_{Q}": "Target"}, inplace=True)
    corr = df.corr()

    ##################################################################
    # Find top_N correlated features
    abs_corr = np.abs(corr["Target"])
    # no_corr = np.where(abs_corr < 0.1)[0]
  
    # top_features = abs_corr.sort_values(ascending=False)[:top_N].index
    # data = corr.loc[top_features, top_features]
    # size_scale=100
    # # corrplot(data, size_scale=150)
    ##################################################################

    # pdb.set_trace()

    sns.ecdfplot(abs_corr)
  
    plt.savefig(f"../figures/correlation_cum_dist.pdf", bbox_inches='tight')


def plot_cum_dist(df_dict, outfile_path):
    
    sns.set_palette("pastel")
    FONTSIZE=12
    for key, df in df_dict.items():
      
        corr = df.corr()
        target_corr = np.abs(corr[f"mean_scl_{key}"])

        sns.ecdfplot(target_corr, label=f"{key}")

    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Absolute Correlation", fontsize=FONTSIZE)
    plt.ylabel("Proportion", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.savefig(outfile_path)
        

    
        
if __name__=="__main__":

    path = "../experiments/Q1_Q3_{Q}/wide_format_analysis/dataframes/train_imputed_cleaned_dataframe_Q1_Q3_{Q}.csv"

    Qs = ["Q4", "Q5", "Q6"]

    df_dict = {}
    for Q in Qs:
        df_dict[Q] = pd.read_csv(path.format(Q=Q))

    outfile_path = "../figures/cum_dist_correlation.pdf"
    plot_cum_dist(df_dict, outfile_path)