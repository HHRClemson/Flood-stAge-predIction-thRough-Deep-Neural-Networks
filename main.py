import argparse

import estimate_depth.estimate_water_depth as depth_estimator
import predict_flooding.predict_flooding as flooding_predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bayes", action="store_true")
    args = parser.parse_args()

    #picture_dataset_path = "./datasets/webcam_images/Shamrock/" # round_to = 5
    picture_dataset_path = "./datasets/webcam_images/ChattahoocheeRiver/" # round_to = 15
    depth_estimator.train_and_predict(picture_dataset_path, round_to=15, bayesian=args.bayes)

