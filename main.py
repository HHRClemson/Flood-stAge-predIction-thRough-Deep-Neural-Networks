import argparse

import estimate_depth.estimate_water_depth as estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bayes", action="store_true")
    args = parser.parse_args()

    dataset_path = "./datasets/webcam_images/Shamrock/"
    #dataset_path = "./datasets/webcam_images/ChattahoocheeRiver/"

    estimator.train_and_predict(dataset_path, round_to=5, bayesian=args.bayes)
