#!/usr/bin/env bash

conda create -n py35 pip python=3.5.5
source activate py35

# pytorch
#conda install pytorch torchvision cuda90 -c pytorch
pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl 

# dependencies
conda install -c anaconda cython numpy  pandas scikit-learn
pip install msgpack
conda install -c conda-forge matplotlib
pip install pynvrtc==8.0 
pip install tensorboardX cupy-cuda90

#spacy
pip install spacy==1.10.1 && python -m spacy.en.download


# prepare dataset
python prepro.py
python semisup_labels.py

