# From-Dark-Matter-to-Galaxies-with-Convolutional-Neural-Networks

This repository contains the codes created to produce this work: https://arxiv.org/abs/1910.07813. The codes are primararily maintained by Jacky H. T. Yip. The paper has been accepted to the NeurIPS Machine Learning and the Physical Sciences Workshop 2019 (acceptance rate: 37%).


General instructions on how to reproduce the results:

| Step | Description | File Path | File Name |
| --- | --- | :---: | :---: |
| 1 | Download raw .hdf5 files from the IllustrisTNG site | /data_related/data_fetching | - |
|



1. Use scripts in  to 
2. Use data_xxx_TNG300-xxx.py in /data_related/data_processing to prepare .npy files of dark matter and galaxy number density fields
3. Use numDen_to_massDen.py in /data_related/data_processing
