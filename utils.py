import pickle
import numpy as np
import torch
import argparse


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

class presDataset(torch.utils.data.Dataset):
    def __init__(self, a, b,c):
        self.pS_array, self.pH_array,self.chat_symptom = a, b, c
    def __getitem__(self, idx):
        sid = self.pS_array[idx]
        hid = self.pH_array[idx]
        chat_symptom_id = self.chat_symptom[idx]
        return sid, hid, chat_symptom_id

    def __len__(self):
        return self.pH_array.shape[0]

def parse_args():
    parser = argparse.ArgumentParser(description='process prescription experiment parameters')
    parser.add_argument('--wandb_name','-n',type=str,default='kdhr',help='every run name')
    parser.add_argument('--run_note','-r',type=str,default='kdhr',help='every run note')
    parser.add_argument('--lr',type=float,default=3e-4,help='lr')
    parser.add_argument('--rec',type=float,default=7e-3,help='weight_decay')
    parser.add_argument('--dropout',type=float,default=0.0,help='dropout')
    parser.add_argument('--batch_size',type=int,default=512,help='batch_size')
    parser.add_argument('--epoch',type=int,default=200,help='epoch')
    parser.add_argument('--dev_ratio',type=float,default=0.2,help='dev_ratio')
    parser.add_argument('--test_ratio',type=float,default=0.2,help='test_ratio')
    parser.add_argument('--chat','-c',action='store_true',default=False,help='component')
    parser.add_argument('--seed',type=int,default=2021,help='seed')
    parser.add_argument('--embedding_dim',type=int,default=64,help='embedding_dim')
    parser.add_argument('--patience',type=int,default=7,help='patience')
    parser.add_argument('--epsilon',type=float,default=1e-13,help='avoid div 0')
    return parser.parse_args()

