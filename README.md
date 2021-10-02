# FloodStat: An End-to-End Flood Detection System Using Ensemble Neural Networks

## Overview
This is the official code repository for the paper `FloodStat: An End-to-End Flood Detection System Using Ensemble Neural Networks` by Windheuser, Pally, Samadi, and Hubig.

It consists out of the following two systems:

1) Estimate the depth of an river based on solely a picture with the code in the directory `./estimate_depth/`.
The CNN and the U-Net is implemented in `./estimate_depth/models/`.
2) The code for the predictions of the floods based on historical data is in the directory `./predict_flooding/`.
The different models, which were compared in the paper, are in `./predict_flooding/models/`.

## Datasets
We were using the data provided by the [United States Geological Survey (USGS)](https://www.usgs.gov/).

For the water depth estimating, we scraped the live webcams of multiple rivers and stored the current image every few minutes.
Our scraped images are available [here](https://syncandshare.lrz.de/getlink/fiMsEgY3zyVpFZCeUxy9Sef7/) and they are free to use for everyone.

For the flood predictions, the USGS is providing various climate data as a csv file for different rivers, which were used as a time-series dataset.


