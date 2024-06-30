import os
import time
import torch
import argparse

from model import SASRec
from utils import *
import pandas as pd
from tqdm import tqdm

class args():
    def __init__(self):
        self.dataset = '../dataset/train_data.txt'
        self.batch_size = 128
        self.lr = 0.0001
        self.maxlen = 50
        self.hidden_units = 200
        self.num_blocks = 4
        self.num_epochs = 101
        self.num_heads  = 4
        self.dropout_rate = 0.1
        self.l2_emb = 0.0
        self.inference_only = False
        self.state_dict_path = None
        self.device = 'cuda:7'

cfg = args()

model = SASRec(53424, 10000, cfg)
model.to("cuda:7")
model.load_state_dict(torch.load('./train_data_result2/SASRec.epoch=101.lr=0.0002.layer=4.head=4.hidden=200.maxlen=50.pth', map_location=torch.device(cfg.device)))
model.eval()

usernum = 0
itemnum = 0
User = defaultdict(list)
user_train = {}
user_valid = {}
user_test = {}

f = open('../dataset/%s.txt' % 'train_data', 'r')
for line in f:
    u, i = line.rstrip().split(',')
    u = int(u)
    i = int(i)
    usernum = max(u, usernum)
    itemnum = max(i, itemnum)
    User[u].append(i)

user_test = defaultdict(list)
for i in tqdm(range(1, usernum+1)):
    j = list(range(1, itemnum+1))
    user_test[i].append(list(set(j).difference(User[i])))
    user_test[i] = user_test[i][0]

with open('./sub.csv', 'ab') as f:
    maxlen_tr = 50
    for i in tqdm(range(1, usernum+1)):
        seq = np.zeros([maxlen_tr], dtype=np.int32)
        idx = maxlen_tr - 1
        for j in reversed(User[i]):
            seq[idx] = j
            idx -= 1
            if idx == -1: break
        item_idx = user_test[i]
        seq = np.array([seq])
        item_idx = np.array(item_idx)
        predictions = -model.predict([i],seq, item_idx)
        predictions = predictions[0]
        a = predictions.argsort()[:10]
        a = a.cpu().numpy()
        r = np.array(item_idx)
        s = r[a]
        u = np.full(shape=10, fill_value=i, dtype=np.int32)
        pre = np.c_[u-1,s-1]
        np.savetxt(f, pre, delimiter=',', fmt='%i')
df = pd.read_csv('./sub.csv',header=None,names=['user_id', 'item_id'])
df.to_csv('./submission_2.csv',index=False)