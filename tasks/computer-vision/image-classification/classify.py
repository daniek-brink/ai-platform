"""
Image classification demo. There are two images in the demos folder that may be used to try this demo.

"""

import os
import argparse
import numpy
from model import DriverDistractionModel
from utils import set_model_params

parser = argparse.ArgumentParser()
arguments = ['model_name', 'path_to_images', 'epochs', 'steps_per_epoch', 'early_stop_patience', 'batch_size', 'learning_rate', 'momentum', 'objective_function',
             'decay', 'training_path', 'validation_path']
for arg in arguments:
       parser.add_argument('--' + arg)
args = parser.parse_args()

model_params = set_model_params(args)

model = DriverDistractionModel(args.model_name, None, None, model_params)
model.build_and_compile_model()
model.load_model()
prediction_data = model.setup_prediction_data(args.path_to_images)
predictions = model.model.predict_generator(prediction_data)
numpy.save(os.path.join(args.path_to_images, 'results.npy'), predictions)
