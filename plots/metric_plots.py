"""Sorry for this ugly code, deadline is getting a bit tight."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rcParams.update({'font.size': 24})

# time points for forecasting
X_POINTS = [3, 6, 9, 12]


def wape_plot_flood():
    columbus_dense = [12, 25, 43, 66.602]
    columbus_cnn = [9, 15, 27, 39.4]
    columbus_lstm = [1.5, 4.1, 5.8, 7.8]
    columbus = [columbus_dense, columbus_cnn, columbus_lstm]

    helen_dense = [14, 17, 29, 39.897]
    helen_cnn = [7, 13, 28, 42.048]
    helen_lstm = [2.3, 4.02, 7.6, 11.926]
    helen = [helen_dense, helen_cnn, helen_lstm]

    sweetwater_dense = [11, 15, 24, 30]
    sweetwater_cnn = [6, 14, 26, 35]
    sweetwater_lstm = [1.35, 3.62, 6.4, 8.06]
    sweetwater = [sweetwater_dense, sweetwater_cnn, sweetwater_lstm]

    datasets = [(columbus, "columbus"), (helen, "helen"), (sweetwater, "sweetwater")]
    plot_bars_flood(datasets, "wape.png")


def plot_depth():
    #columbus_segmentation = 0.6918
    #columbus_no_segmentation = 3.6228
    columbus_segmentation = 0.0654
    columbus_no_segmentation = 0.3310

    #sweetwater_segmentation = 1.028
    #sweetwater_no_segmentation = 1.5540
    sweetwater_segmentation = 0.0035
    sweetwater_no_segmentation = 0.0049

    segmentation = [columbus_segmentation, sweetwater_segmentation]
    non_segmentation = [columbus_no_segmentation, sweetwater_no_segmentation]
    X = np.arange(2)

    plt.bar(X - 0.2, segmentation, color="blue", width=0.4)
    plt.bar(X + 0.2, non_segmentation, color="orange", width=0.4)

    for x, y in [(0 - 0.2, segmentation[0]), (0 + 0.2, non_segmentation[0]),
                 (1 - 0.2, segmentation[1]), (1 + 0.2, non_segmentation[1])]:

        plt.text(x, y + 0.01, str(y), fontweight="bold", ha="center")

    plt.xticks(X, ["Columbus", "Sweetwater Creek"])

    blue_patch = mpatches.Patch(color='blue', label='with U-Net')
    orange_patch = mpatches.Patch(color='orange', label='without U-Net')
    plt.legend(handles=[blue_patch, orange_patch])
    plt.ylabel("Mean Absolute Error (MAE) [feet]")
    plt.axis([-0.5, 1.5, 0, 0.4])

    plt.show()


def plot_bars_flood(datasets, suffix):
    colors = ["blue", "orange", "red"]

    width = 0.8

    for dataset, ds_name in datasets:
        bars = []
        for i, t in enumerate(X_POINTS):
            curr_time = []
            for model in dataset:
                curr_time.append(model[i])
            bars.append(curr_time)

        for i, bar in enumerate(bars):
            for j, val in enumerate(bar):
                plt.xticks(ticks=X_POINTS, label="min")
                plt.bar(X_POINTS[i] + (j - 1) * width, val, width, color=colors[j])
                plt.text(X_POINTS[i] + (j - 1) * width, val + 0.75, str(val) + '%',
                         fontweight='bold', ha='center')

        blue_patch = mpatches.Patch(color='blue', label='Dense')
        orange_patch = mpatches.Patch(color='orange', label='CNN')
        red_patch = mpatches.Patch(color='red', label='LSTM')
        plt.legend(handles=[blue_patch, orange_patch, red_patch])

        plt.xlabel("Forecast time [h]")
        plt.ylabel("Weighted Average Percentage Error (WAPE)")
        name = ds_name + "-" + suffix
        #plt.savefig(name)
        plt.show()


if __name__ == "__main__":
    #wape_plot_flood()
    plot_depth()
