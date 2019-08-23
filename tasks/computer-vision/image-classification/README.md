# Image Classification

This model classifies an image of a driver as one of the following:

- c0: safe driving
- c1: texting - right
- c2: talking on the phone - right
- c3: texting - left
- c4: talking on the phone - left
- c5: operating the radio
- c6: drinking
- c7: reaching behind
- c8: hair and makeup
- c9: talking to passenger

In order to train this model, do the following:

1. Run `bash setup_environment.sh` from this directory.
2. Run `mlflow run . -e  train.py -P training_path=<path_to_training_imgs> -P validation_path=<path_to_validation_imgs> -P model_name=<your_model_name> -P epochs=10`
3. See the MLproject file for additional model training parameters that can be set.

In order to predict using this model, do the following:

1. Train the model by following the steps above.
2. Run ` mlflow run . -e classify -P model_name=<your_model_name> -P path_to_image=<path to image>`