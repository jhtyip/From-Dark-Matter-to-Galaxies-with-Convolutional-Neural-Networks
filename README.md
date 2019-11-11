# From-Dark-Matter-to-Galaxies-with-Convolutional-Neural-Networks

This repository contains the codes created to produce this work: https://arxiv.org/abs/1910.07813. The codes are primararily maintained by Jacky H. T. Yip. The paper has been accepted to the NeurIPS Machine Learning and the Physical Sciences Workshop 2019 (acceptance rate: 37%).

General instructions on how to reproduce the results[1]:

| Step | Description | File Path | File Name |
| :---: | --- | :---: | :---: |
| 1 | Download raw snapshot .hdf5 files from the IllustrisTNG site | /data_related/data_fetching | - |
| 2 | Prepare .npy files of dark matter and galaxy number density fields | /data_related/data_processing | data_xxx_TNG300-xxx.py |
| 3 | Convert the dark matter number density field to mass density field | /data_related/data_processing | numDen_to_massDen.py |
| 4 | Train phases of the cascade CNNs individually with selected hyperparameters | /training | main.py |
| 5 | Generate the prediction field with the trained model | /tools | npyGen.py |
| 6 | Prepare the galaxy number density field from the HOD algorithm | /HOD | HOD.py |
| 7 | Calculate and plot power spectra and bispectra[2] | /tools/PowSpec_and_BiSpec | - |
| 8 | Further analysis on the outputs | /tools/cube_analysis | - |

[1] With Python 3.5.5 and PyTorch 0.4.1  
[2] More on getting the bispectra: https://github.com/franciscovillaescusa/Pylians
