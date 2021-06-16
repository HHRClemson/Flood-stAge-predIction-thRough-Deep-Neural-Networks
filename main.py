import argparse

import predict_depth.predict_water_depth as predicter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bayes", action="store_true")
    args = parser.parse_args()

    #dataset_path = "./datasets/new_data/Shamrock/"
    dataset_path = "./datasets/RockyCreek/"

    predicter.train_and_predict(dataset_path, bayesian=args.bayes)
