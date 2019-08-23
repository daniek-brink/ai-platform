"""
Driver distraction model class
"""

import os
import mlflow
import mlflow.keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Reshape, Flatten, Concatenate
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from openpose_keras.model import get_training_model
from constants import NUM_CLASSES, PRE_TRAINED_MODEL_PATH, TRAINED_MODELS_DIR, CHECKPOINTS_DIR, TENSORBOARD_LOG_DIR


class DriverDistractionModel():

    """
    Class that instantiates a model to predict what a driver is doing, i.e. paying attention, looking at phone, etc.
    """

    def __init__(self, model_file_name, path_to_training_data, path_to_validation_data, model_params):
        """
        Initialize the model class
        :param model_file_name: File name of the model located in the trained_models directory. When training, this file is created.
        :param path_to_training_data: Path to the directory that contains the training data.
        :param path_to_validation_data: Path to the directory that contains the evaluation data.
        :param model_params: Dictionary that contains the model parameters for training.
        """

        self.path_to_training_data = path_to_training_data
        self.path_to_validation_data = path_to_validation_data
        self.model_file_name = model_file_name
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.prediction_generator = None

        # Openpose parameters
        self.weight_decay = 5e-4
        self.openpose_param_1 = 38
        self.openpose_param_2 = 19
        self.openpose_stages = 6

        # Training parameters
        self.loss_metrics = ['accuracy']
        self.model_params = model_params

    def fit(self):
        """
        Fit the model and saves it using MLFlow.
        """

        self.setup_training_data()
        self.build_and_compile_model()

        # Create early stopping callback
        cb_early_stopper = EarlyStopping(monitor='val_loss', patience=self.model_params['early_stop_patience'])

        # Create model checkpoint callback
        cb_checkpointer = ModelCheckpoint(filepath=os.path.join(CHECKPOINTS_DIR, self.model_file_name), monitor='val_loss',
                                          save_best_only=True, mode='auto')

        # Build tensorboard logging callback
        cb_tensorboard = TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=0, batch_size=self.model_params['batch_size'],
                                     write_graph=True, write_grads=False)

        with mlflow.start_run():
            mlflow.log_param("epochs", self.model_params['epochs'])
            mlflow.keras.log_model(self.model, "models")
            self.model.fit_generator(self.train_generator,
                                     steps_per_epoch=self.model_params['steps_per_epoch'],
                                     epochs=self.model_params['epochs'],
                                     validation_data=self.validation_generator,
                                     validation_steps=self.model_params['steps_per_epoch'],
                                     callbacks=[cb_checkpointer, cb_early_stopper, cb_tensorboard])

        mlflow.keras.save_model(self.model, os.path.join(TRAINED_MODELS_DIR, self.model_file_name), conda_env='conda.yaml')

    def build_and_compile_model(self):
        """
        Setup the model architecture.
        """

        open_pose_model = get_training_model(self.weight_decay, self.openpose_param_1, self.openpose_param_2, stages=self.openpose_stages)
        open_pose_model.load_weights(PRE_TRAINED_MODEL_PATH)

        # This may seem unnecessary, but the original model is a keras model, whereas the rest of this code
        # is compatible with a tensorlfow.keras model, hence the conversion.
        config = open_pose_model.get_config()
        weights = open_pose_model.get_weights()
        pre_trained_model = Model.from_config(config)
        pre_trained_model.set_weights(weights)

        x = Dense(NUM_CLASSES, activation='softmax', name='first_dense_layer')(pre_trained_model.get_layer('concatenate_5').output)
        x = Reshape((NUM_CLASSES*32*32,))(x)
        x = Dense(NUM_CLASSES, name='final_dense_layer', activation='softmax')(x)
        self.model = Model(pre_trained_model.input[0], x)

        for i in range(0, len(self.model.layers)):
            if self.model.layers[i].name not in ['first_dense_layer', 'final_dense_layer']:
                self.model.layers[i].trainable = False

        sgd = optimizers.SGD(lr=self.model_params['learning_rate'], decay=self.model_params['decay'],
                             momentum=self.model_params['momentum'], nesterov=True)
        self.model.compile(optimizer=sgd, loss=self.model_params['objective_function'], metrics=self.loss_metrics)

    def load_model(self):
        """
        Load the model.
        """
        self.model = mlflow.keras.load_model(os.path.join(TRAINED_MODELS_DIR, self.model_file_name))

    def setup_training_data(self):
        """
        Set up data for training.
        """

        data_generator = ImageDataGenerator()

        self.train_generator = data_generator.flow_from_directory(self.path_to_training_data, batch_size=self.model_params['batch_size'],
                                                                  shuffle=True, class_mode='categorical')
        self.validation_generator = data_generator.flow_from_directory(self.path_to_validation_data, batch_size=self.model_params['batch_size'],
                                                                       class_mode='categorical')

    @staticmethod
    def setup_prediction_data(path):
        """
        Setup data for prediction.
        :return Test generator for use with model.predict_generator
        """
        data_generator = ImageDataGenerator()
        test_generator = data_generator.flow_from_directory(path, batch_size=1, shuffle=False, class_mode=None)
        return test_generator
