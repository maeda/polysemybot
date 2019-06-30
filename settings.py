import os

import torch
from dotenv import load_dotenv
load_dotenv(verbose=True)

print("NUM CORES: " + str(torch.get_num_threads()))
torch.set_num_threads(torch.get_num_threads())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.environ.get("BASE_DIR")
TRAINING_DATA_DIR = os.environ.get("TRAINING_DATA_DIR")
SAVE_DATA_DIR = os.environ.get("SAVE_DATA_DIR")
