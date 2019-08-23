#!/usr/bin/env bash

# Important: run this script from the "image-classification" directory.

git clone https://github.com/cchamber/openpose_keras.git
mkdir openpose_keras_models/
wget --directory-prefix openpose_keras_models/ https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5