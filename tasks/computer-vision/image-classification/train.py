"""
Training demo.

In order to use, download the data set from https://www.kaggle.com/jerrinv/driver-distraction/data.
Next run the script setup_training_data.sh.
To run this script, supply the path to the training images, the validation images and the file name of the new model, for example:
`python demos/train.py ~/Downloads/kaggle/imgs/train/ ~/Downloads/kaggle/imgs/validation/ some_new_model`
"""

#TODO: write setup_training_data script

from model import DriverDistractionModel
import sys

# Unpack the arguments to variables
keys = 'path_to_training_images', 'path_to_validation_images', 'model_path', \
       'batch_size', 'epochs', 'number_of_steps'
args = {key: value for key, value in zip(keys, sys.argv[1:])}
train_path = args['path_to_training_images']
validation_path = args['path_to_validation_images']
model_name = args['model_name']

model = DriverDistractionModel(model_name, train_path, validation_path)
model.fit()

