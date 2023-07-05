import xray_data
import uae_main
import torch
from argparse import ArgumentParser
import json

if __name__ == "__main__":
    model_name = "Trial_2_UPAE"

    # Create Argument parser accept multiple JSON Files
    json_files = ArgumentParser()
    json_files.add_argument('json_files', nargs='+')

    args = json_files.parse_args()
    for model_name in args.json_files:
        print(f"Training Model {model_name}")
        # Get JSON file from models/model_parameters/
        model_parameters = json.load(open(f'models/model_parameters/{model_name}.json'))

        # Set up the arguments for the model from the JSON file
        parser = ArgumentParser()
        for key, value in model_parameters.items():
            parser.add_argument(f"--{key}", dest=key, type=type(value), default=value)

        # Parse the arguments
        opt = parser.parse_args(args=[])

        # Trains the model
        uae_main.train(opt)
        uae_main.test_for_xray(opt, plot=True, plot_name=model_name)