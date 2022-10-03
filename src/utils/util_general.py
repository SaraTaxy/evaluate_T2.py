import argparse
from pathlib import Path
import os
import torch
import numpy as np
import random
import requests


def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()


def get_args_2():
    '''
    Read 2 configuration files for comparison (ex. diversity_analysis.py)
    '''
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f1", "--cfg_file_1", help="Path of Configuration File 1", type=str)
    parser.add_argument("-f2", "--cfg_file_2", help="Path of Configuration File 2", type=str)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()


def seed_all(seed):
    if not seed:
        seed = 0
    print("Using Seed : ", seed)

    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Empty and create direcotory
def create_dir(dir):
    if not os.path.exists(dir):
        Path(dir).mkdir(parents=True, exist_ok=True)


def delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass


def notify_IFTTT(info):
    private_key = "bZvH5hifEuGRzZvCJSCBTj"
    url = "https://maker.ifttt.com/trigger/Notification/with/key/" + private_key
    requests.post(url, data={'value1': str(info)})
