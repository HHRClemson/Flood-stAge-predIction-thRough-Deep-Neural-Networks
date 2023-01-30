<h1 align="center">FAIRDNN <br>
Flood stAge predIction thRough Deep Neural Networks
</h1>

<h3 align="center">
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)<br>
</h3>

## Overview
This is the official code repository for the paper [An End-to-End Flood Detection System Using Deep Neural Networks (2023)](https://doi.org/10.1029/2022EA002385) by Windheuser, Karanjit, Pally, Samadi, and Hubig.

It consists out of the following two systems:

1) Estimate the depth of an river based on solely a picture with the code in the directory `./estimate_depth/`.
The CNN and the U-Net is implemented in `./estimate_depth/models/`.
2) The code for the predictions of the gauge height based on historical data is in the directory `./predict_flooding/`.
The different models, which were compared in the paper, are in `./predict_flooding/models/`.

If you find our code or paper, published at the AGU Earth & Space Science, useful we encourage you to cite our paper. BibTeX:

`@article{windheuserend,
  title={An End-to-End Flood Stage Prediction System Using Deep Neural Networks},
  author={Windheuser, L and Karanjit, R and Pally, R and Samadi, S and Hubig, NC},
  journal={Earth and Space Science},
  pages={e2022EA002385},
  publisher={Wiley Online Library}
}`

## Datasets
We used the data provided by the [United States Geological Survey (USGS)](https://www.usgs.gov/).

For the water depth estimation, we scraped the live webcams of multiple rivers and stored the current image every few minutes.
Our scraped images are available [here](https://syncandshare.lrz.de/getlink/fiMsEgY3zyVpFZCeUxy9Sef7/) and they are free to use for everyone.

For the flood predictions, the USGS provides various climate data as a csv file for different rivers, which were used as a time-series dataset.

## Authors

* **Leon Windheuser** 
* **Rishav Karanjit**
* **Rakshit Pally** 
* **Dr. Vidya Samadi** 
* **Dr. Nina Hubig** 

## Cite Our Work
If you are interested in our work from an academic standpoint, please cite our paper:

```bibtex
@article{windheuserend,
  title={An End-to-End Flood Stage Prediction System Using Deep Neural Networks},
  author={Windheuser, L and Karanjit, R and Pally, R and Samadi, S and Hubig, NC},
  journal={Earth and Space Science},
  pages={e2022EA002385},
  publisher={Wiley Online Library}
}
```

