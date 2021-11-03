# On the use of Deep Generative Models for "Perfect" Prognosis Climate Downscaling

This repository contains the code to reproduce the use-case experiment of the paper *On the use of Deep Generative Models for "Perfect" Prognosis Climate Downscaling* published at Tackling Climate Change with Machine Learning Workshop (NeurIPS 2021).

### Introduction

Deep Learning has recently emerged as a "perfect" prognosis downscaling technique to compute high-resolution fields from large-scale coarse atmospheric data. Despite their promising results to reproduce the observed local variability, they are based on the estimation of independent distributions at each location, which leads to deficient spatial structures, especially when downscaling precipitation. This study proposes the use of generative models to improve the spatial consistency of the high-resolution fields, very demanded by some sectoral applications (e.g., hydrology) to tackle climate change.

To illustrate these points we develop a simple use-case of Perfect Prognosis Downscaling over Europe using a Generative Model, more specifically a Conditional Variational Autoencoder (CVAE):

![CVAE model](https://github.com/jgonzalezab/CVAE-PP-Downscaling/blob/main/CVAE/figures/figurePaper.jpg)

### Installation
A [Dockerfile](https://github.com/jgonzalezab/CVAE-PP-Downscaling/blob/main/Dockerfile) is available with all the libraries needed to run the experiment

### Download and preprocess data

The notebook [preprocessData.ipynb](https://github.com/jgonzalezab/CVAE-PP-Downscaling/blob/main/CVAE/preprocessData.ipynb) is available with the code and instructions to download and preprocess the data. To download the data an account in [UDG-TAP](http://meteo.unican.es/udg-tap/home) may be required.

### Train the model and compute stochastic samples

By running the [runModel.ipynb](https://github.com/jgonzalezab/CVAE-PP-Downscaling/blob/main/CVAE/runModel.ipynb) notebook, the CVAE model can be trained. A pre-trained model is also available for the user to directly generate conditioned samples.
