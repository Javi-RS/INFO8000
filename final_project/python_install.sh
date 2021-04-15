#!/bin/bash

pip install virtualenv
virtualenv foss_groupF
source foss_groupF/bin/activate

pip install base64
pip install flask==1.1.2
pip install matplotlib==3.3.2
pip install numpy==1.19.5
pip install opencsv==4.2.0.34
pip install opencv-contrib-python==4.5.1.48
pip install os==0.2.14
pip install pandas==1.1.3
#pip install PIL==1.1.6
pip install pillow==8.2.0 #Pillow is a maintained fork of PIL
pip install seaborn==0.11.0
pip install scikit-learn==0.22
pip install werkzeug.utils==1.0.1


