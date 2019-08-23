"""
Training demo.

In order to use, download the data set from https://www.kaggle.com/jerrinv/driver-distraction/data.
Next run the script setup_training_data.sh.
To run this script, supply the path to the training images, the validation images and the file name of the new model, for example:
`python demos/train.py --training_path ~/Downloads/kaggle/imgs/train/ --validation_path ~/Downloads/kaggle/imgs/validation/ --model_name some_new_model --epochs 1 --batch_size 2 --steps_per_epoch 2`
"""

#TODO: write setup_training_data script

from model import DriverDistractionModel
from utils import set_model_params
import argparse

parser = argparse.ArgumentParser()
arguments = ['epochs', 'steps_per_epoch', 'early_stop_patience', 'batch_size', 'learning_rate', 'momentum', 'objective_function',
             'decay', 'training_path', 'validation_path', 'model_name']
for arg in arguments:
       parser.add_argument('--' + arg)

args = parser.parse_args()

model_params = set_model_params(args)

model = DriverDistractionModel(args.model_name, args.training_path, args.validation_path, model_params)
model.fit()

