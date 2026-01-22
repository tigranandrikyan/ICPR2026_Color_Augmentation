# Fancy GNG

## Overview


Deep artificial neural networks require extensive training datasets to achieve optimal learning outcomes; however, collecting such datasets is often resource-intensive and laborious. Data augmentation mitigates this challenge by artificially expanding the training corpus through label-preserving transformations, thereby enhancing model generalization. Although prior research has extensively benchmarked general data augmentation methods to improve Convolutional Neural Network (CNN) performance, the specific impact of advanced color augmentation techniques remains underexplored. This code extends the widely adopted Fancy PCA method by proposing Fancy GNG, an advanced color augmentation technique that utilizes clustering of color spaces with Growing Neural Gas. The experimental results from the related paper ("Fancy Growing Neural Gas for Color Augmentation") demonstrate that Fancy GNG yields substantial performance improvements, offering novel insights into optimizing color augmentation strategies for CNN-based applications.

## Application Description

This repository contains a **Streamlit-based web application** that enables interactive exploration of color augmentation methods, with a particular focus on Fancy GNG.

The application allows users to:

* Apply color augmentation techniques to images
* Capture images directly within the web application
* Upload and process local image files
* Adjust and experiment with augmentation parameters
* Generate visualizations illustrating the effects of the applied augmentations
* Download augmentated images

The app is intended both as a visualization tool for reproducing and understanding the proposed method, and as a practical tool for experimenting with advanced color augmentation techniques.

## Live Demo
The Streamlit application is publicly available at:
https://fancygng.streamlit.app/

For further question, please contact: tigran.andrikyan@uni-luebeck.de