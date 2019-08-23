"""
Image classification demo. There are two images in the demos folder that may be used to try this demo.

"""

from model import DriverDistractionModel
import cv2
import numpy
import argparse
from utils import set_model_params

parser = argparse.ArgumentParser()
arguments = ['model_name', 'path_to_image', 'epochs', 'steps_per_epoch', 'early_stop_patience', 'batch_size', 'learning_rate', 'momentum', 'objective_function',
             'decay', 'training_path', 'validation_path']
for arg in arguments:
       parser.add_argument('--' + arg)
args = parser.parse_args()

model_params = set_model_params(args)

model = DriverDistractionModel(args.model_name, None, None, model_params)
model.build_and_compile_model()
model.load_model()

img = cv2.resize(cv2.imread(args.path_to_image), (0, 0), fx=256, fy=256, interpolation=cv2.INTER_CUBIC)

predictions = model.predict(numpy.expand_dims(img, axis=0))
print(predictions)