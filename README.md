# From-Dark-Matter-to-Galaxies-with-Convolutional-Neural-Networks

This repository contains the codes created to produce this work: https://arxiv.org/abs/1910.07813. The codes are primararily maintained by Jacky H. T. Yip. The paper has been accepted to the NeurIPS Machine Learning and the Physical Sciences Workshop 2019 (acceptance rate: 37%).


General instructions on how to reproduce the results:

| Step | Description | File Path | File Name |
| :---: | --- | :---: | :---: |
| 1 | Download raw snapshot .hdf5 files from the IllustrisTNG site | /data_related/data_fetching | - |
| 2 | Prepare .npy files of dark matter and galaxy number density fields | /data_related/data_processing | data_xxx_TNG300-xxx.py |
| 3 | Convert the dark matter number density field to mass density field | /data_related/data_processing | numDen_to_massDen.py |
