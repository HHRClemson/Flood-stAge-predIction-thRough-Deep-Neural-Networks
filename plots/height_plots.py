import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rcParams.update({'font.size': 20})


def violin_plot(prefix):
    columbus = pd.read_csv(prefix + "chattahoochee-columbus.csv")
    helen = pd.read_csv(prefix + "chattahoochee-helen.csv")
    sweetwater = pd.read_csv(prefix + "sweetwater-creek.csv")

    ax = sns.violinplot(data=[columbus["height"],
                         helen["height"],
                         sweetwater["height"]],
                         palette="muted",
                         orient="h")

    ax.set_yticklabels(["Columbus", "Helen", "Sweetwater Creek"],
                        rotation=90, va="center")
    ax.set_xlabel("Height [feet]")

    plt.show()


def distribution_plot(prefix):
    columbus = pd.read_csv(prefix + "chattahoochee-columbus.csv")
    helen = pd.read_csv(prefix + "chattahoochee-helen.csv")
    sweetwater = pd.read_csv(prefix + "sweetwater-creek.csv")
     
    ax = sns.displot(columbus["height"], color="blue", palette="muted")
    ax.set(xlabel="Height [feet]")
    plt.savefig("columbus-height.png",bbox_inches='tight')

    ax = sns.displot(helen["height"], color="green", palette="muted")
    ax.set(xlabel="Height [feet]")
    plt.savefig("helen-height.png",bbox_inches='tight')

    ax = sns.displot(sweetwater["height"], color="orange", palette="muted")
    ax.set(xlabel="Height [feet]")
    plt.savefig("sweetwater-height.png",bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    path_prefix = "../datasets/time_series/"
    #violin_plot(path_prefix)
    distribution_plot(path_prefix)
