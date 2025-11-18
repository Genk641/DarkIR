import numpy as np
import os, sys
from tqdm import tqdm
from options.options import parse
import argparse

parser = argparse.ArgumentParser(description="Script for testing")
parser.add_argument('-p', '--config', type=str, default='./options/test/LOLBlur.yml', help = 'Config file of testing')
args = parser.parse_args()

# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need
path_options = args.config
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= "0" # you need to fix this before importing torch

# PyTorch library
import torch
import torch.optim

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import create_path_models
from utils.test_utils import eval_model
from ptflops import get_model_complexity_info

#parameters for saving model
PATH_MODEL= create_path_models(opt['save'])


def load_model(model, path_weights):

    map_location = 'cpu'
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
    # print(checkpoints.keys())
    # sys.exit()
    weights = checkpoints['params']
    weights = {'module.' + key: value for key, value in weights.items()}

    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
    print('Network complexity: ' ,macs, params)

    model.load_state_dict(weights)
    print('Loaded weights correctly')
    
    return model

def run_evaluation(device):

    test_loader, _ = create_test_data(rank=0, world_size=1, opt=opt['datasets'])
    model, _, _ = create_model(opt['network'], rank=device)

    model = load_model(model, opt['save']['path'])
    metrics_eval = {}

    model.eval()
    metrics_eval, _ = eval_model(model, test_loader, metrics_eval, device=device, eta=True)

    if type(next(iter(metrics_eval.values()))) == dict:
        for key, metric_eval in metrics_eval.items():
            print(f" \t {key} --- PSNR: {metric_eval['valid_psnr']}, SSIM: {metric_eval['valid_ssim']}, LPIPS: {metric_eval['valid_lpips']}")
    else:
        print(f" \t {opt['datasets']['name']} --- PSNR: {metrics_eval['valid_psnr']}, SSIM: {metrics_eval['valid_ssim']}, LPIPS: {metrics_eval['valid_lpips']}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_evaluation(device)

if __name__ == '__main__':
    main()
