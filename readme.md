Classification models for predicting the bioactivity of pan-TRK inhibitors and SAR analysis
This repository contains the source code and the data.

Graphical abstract


Setup and dependencies
Dependencies:
python 3.7
RDkit = 2019.09.3
Numpy = 1.19.0
Pandas = 0.24.2
Torch = 1.13.0
Pydotplus = 2.0.2
scikit-learn = 0.20.3

Data and usage
The data sets are provided as .csv files in a directory called 'data',which is the original collection of processed molecules. It also needs to be processed by fingerprint, kmeanscluster to meet the requirements of the model input data.