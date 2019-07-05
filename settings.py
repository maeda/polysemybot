import os, logging

import torch
from dotenv import load_dotenv
load_dotenv(verbose=True)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("NUM CORES: " + str(torch.get_num_threads()))
torch.set_num_threads(torch.get_num_threads())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cores = torch.get_num_threads()

BASE_DIR = os.environ.get("BASE_DIR")
TRAINING_DATA_DIR = os.environ.get("TRAINING_DATA_DIR")
SAVE_DATA_DIR = os.environ.get("SAVE_DATA_DIR")
