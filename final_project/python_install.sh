#!/bin/bash

pip install virtualenv
virtualenv foss_groupF
source foss_groupF/bin/activate

pip install flask==1.1.2
pip install werkzeug.utils==1.0.1
pip install scikit-learn==0.21.2
pip install numpy==1.18.2
pip install pandas==1.0.3
pip install seaborn==0.10.0
pip install PIL==1.1.6
pip install os==0.2.14
pip install base64
pip install opencsv==4.2.0.34
pip install opencv-contrib-python


