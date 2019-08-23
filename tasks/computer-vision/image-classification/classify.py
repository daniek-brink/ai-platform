"""
Image classification demo. There are two images in the demos folder that may be used to try this demo.

"""

from model import DriverDistractionModel
from constants import DEFAULT_EXISTING_MODEL_NAME
import sys
import cv2
import numpy

# TODO: see if i can simplify this
# Unpack the arguments to variables
keys = 'test_img', 'model_name'
args = {key: value for key, value in zip(keys, sys.argv[1:])}
test_img = str(args['test_img'])
model_name = args['model_name']


model = DriverDistractionModel(model_name, None, None)
model.build_and_compile_model()
model.load_model()

img = cv2.resize(cv2.imread(test_img), (0, 0), fx=256, fy=256, interpolation=cv2.INTER_CUBIC)

predictions = model.predict(img[numpy.newaxis, :, :, :])
print(predictions)