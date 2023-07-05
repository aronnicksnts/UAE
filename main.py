import xray_data
import uae_main
import torch
from argparse import ArgumentParser
import json

model_name = "Trial_2_UPAE"
# Get JSON file from models/model_parameters/
model_parameters = json.load(open(f'models/model_parameters/{model_name}.json'))

# Set up the arguments for the model from the JSON file
parser = ArgumentParser()
for key, value in model_parameters.items():
    parser.add_argument(f"--{key}", dest=key, type=type(value), default=value)
    print("key: ", key, "value: ", value)

# Parse the arguments
opt = parser.parse_args(args=[])

uae_main.train(opt)
uae_main.test_for_xray(opt, plot=True)