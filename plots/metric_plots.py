"""Sorry for this ugly code, deadline is getting a bit tight."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rcParams.update({'font.size': 24})

# time points for forecasting
X_POINTS = [3, 6, 9, 12]


def mae_plot():
    pass


def mse_plot():
    pass


def rmse_plot():
    pass


def wape_plot():
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

    datasets = [columbus, helen, sweetwater]
    datasets = [(columbus, "columbus"), (helen, "helen"), (sweetwater, "sweetwater")]
    plot_bars(datasets, "wape.png")


def plot_bars(datasets, suffix):
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
    mae_plot()
    mse_plot()
    rmse_plot()
    wape_plot()
