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
    columbus_dense = [4.259, 5.967, 19.728, 38.796]
    columbus_cnn = [6.222, 8.134, 15.103, 24.102]
    columbus_lstm = [3.17, 5.505, 6.802, 12.8]
    columbus = [columbus_dense, columbus_cnn, columbus_lstm]

    helen_dense = [6.746, 12.339, 24.483, 37.318]
    helen_cnn = [11.223, 24.484, 37.318, 46.048]
    helen_lstm = [2.3, 4.02, 7.6, 11.926]
    helen = [helen_dense, helen_cnn, helen_lstm]

    sweetwater_dense = [7.631, 10.054, 20.053, 33.128]
    sweetwater_cnn = [16.177, 22.901, 25.237, 28.016]
    sweetwater_lstm = [4.92, 6.153, 15.288, 21.277]
    sweetwater = [sweetwater_dense, sweetwater_cnn, sweetwater_lstm]

    datasets = [(columbus, "columbus"), (helen, "helen"), (sweetwater, "sweetwater")]
    plot_bars_flood(datasets, "wape.png")


def mae_plot_flood():
    columbus_dense = [0.199, 0.302, 0.373, 0.403]
    columbus_cnn = [0.224, 0.361, 0.395, 0.406]
    columbus_lstm = [0.166, 0.245, 0.293, 0.304]
    columbus = [columbus_dense, columbus_cnn, columbus_lstm]

    helen_dense = [0.098, 0.144, 0.177, 0.211]
    helen_cnn = [0.093, 0.136, 0.181, 0.223]
    helen_lstm = [0.083, 0.133, 0.156, 0.181]
    helen = [helen_dense, helen_cnn, helen_lstm]

    sweetwater_dense = [0.077, 0.107, 0.138, 0.175]
    sweetwater_cnn = [0.087, 0.121, 0.128, 0.149]
    sweetwater_lstm = [0.027, 0.065, 0.103, 0.102]
    sweetwater = [sweetwater_dense, sweetwater_cnn, sweetwater_lstm]

    datasets = [(columbus, "columbus"), (helen, "helen"), (sweetwater, "sweetwater")]
    plot_bars_flood(datasets, "mae.png")


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
                plt.text(X_POINTS[i] + (j - 1) * width, val + 0.005, str(round(val, 2)) + '%',
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
    wape_plot_flood()
    #mae_plot_flood()
    #plot_depth()
