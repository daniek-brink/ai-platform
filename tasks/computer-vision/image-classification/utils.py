"""
Common utilities
"""


def set_model_params(args):
    """
    Get the model parameters for training
    :param args: Argparse arguments to set the model parameters
    :return: Dictionary of model parameters.
    """

    # Define default model parameters
    model_params = {
       'epochs': int(args.epochs) if args.epochs else 15,
       'steps_per_epoch': int(args.steps_per_epoch) if args.steps_per_epoch else 400,
       'early_stop_patience': int(args.early_stop_patience) if args.early_stop_patience else 3,
       'batch_size': int(args.batch_size) if args.batch_size else 30,
       'learning_rate': float(args.learning_rate) if args.learning_rate else 0.01,
       'momentum': float(args.momentum) if args.momentum else 0.9,
       'objective_function': args.objective_function if args.objective_function else 'categorical_crossentropy',
       'decay': float(args.decay) if args.decay else 1e-6}
    return model_params
