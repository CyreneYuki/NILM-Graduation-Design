import argparse
import torch
from run_model import test_model_ae
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataloador import prompt_learn_load_data
import torch.nn as nn

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='classification', help="classification/regression")
parser.add_argument("--dataset", type=str, default='redd', help="redd/ukdale")
parser.add_argument('--save_dir', type=str, default='./models_save', help='The directory to save the trained models')
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument('--sample_rate', type=int, default=256, help='The sample interval of trainset')
parser.add_argument('--window_len', type=int, default=1024, help='The len of window, microwave=256')
parser.add_argument('--test_house', type=int, default=3, help='The num of house to test')
parser.add_argument('--applist', type=list, default=['fridge', 'dishwasher', 'microwave', 'washingmachine'],
                    help='washingmachine, fridge, dishwasher, microwave')
parser.add_argument('--test_app', type=str, default='dishwasher',
                    help='washingmachine, fridge, dishwasher, microwave')

opt = parser.parse_args()
print(opt)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(t1, t2):

    'loading data and model'
    data_path = './dataset/'+opt.dataset+'/processed/house_'
    encoder = torch.load(opt.save_dir + '/'+opt.dataset+ '/'+opt.test_app+'/Encoder.pth').to(device)
    decoder = torch.load(opt.save_dir + '/'+opt.dataset+ '/'+opt.test_app+'/Decoder.pth').to(device)

    AE_model = nn.Sequential(encoder, decoder)

    'testing'
    test_iterator = prompt_learn_load_data(data_path=data_path,
                                           window_len=opt.window_len,
                                           house_select=opt.test_house,
                                           applist=opt.applist,
                                           app=opt.test_app,
                                           batch_size=opt.batch_size,
                                           sample_rate=opt.sample_rate,)

    agg, truth_power, pred, truth_onoff = test_model_ae(model=AE_model,
                                                  iterator=test_iterator,
                                                  device=device,)

    if opt.mode == 'classification':
        'processing result'
        pred_onoff = torch.sigmoid(pred)
        pred_onoff[pred_onoff > 0.5] = 1
        pred_onoff[pred_onoff <= 0.5] = 0

        'evaluating'
        acc = accuracy_score(truth_onoff, pred_onoff)
        p = precision_score(truth_onoff, pred_onoff)
        r = recall_score(truth_onoff, pred_onoff)
        f1 = f1_score(truth_onoff, pred_onoff)
        print(f'acc: {acc}\trecall: {r}\tprecision: {p}\tF1: {f1}')

        pred_onoff = pred_onoff[t1:t2]

    'visualization'
    agg, truth_power, pred, truth_onoff = agg[t1:t2], truth_power[t1:t2], pred[t1:t2], truth_onoff[t1:t2]
    max_power = agg.max().numpy()
    x_axis = np.linspace(0, len(agg), len(agg)).astype(int)
    plt.plot(x_axis, agg)
    plt.plot(x_axis, truth_power)

    if opt.mode == 'regression':
        plt.plot(x_axis, pred)
        plt.legend(['Agg', 'Truth Power', 'Pred Power'])

    elif opt.mode == 'classification':
        plt.legend(['Agg', 'Truth Power'])
        plt.fill_between(x_axis, np.zeros_like(truth_power), np.ones_like(truth_power) * max_power,
                         where=truth_onoff == 1, color='tomato', alpha=0.5)
        plt.fill_between(x_axis, np.zeros_like(truth_power), np.ones_like(truth_power) * max_power,
                         where=pred_onoff == 1, color='dodgerblue', alpha=0.5)
    plt.show()


test(0, 20000)