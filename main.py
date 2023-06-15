from utils import *
from model import *
# from model_compara import Compare
# from model_SMGCN import SMGCN
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from pytorchtools import EarlyStopping
import time
import wandb
from tqdm import tqdm

args = parse_args()


wandb.init(project="khdr",
           name=args.wandb_name,
           notes=args.run_note,
           config={
               "lr": args.lr,
               "rec": args.rec,
               "dropout": args.dropout,
               "batch_size": args.batch_size,
               "epoch": args.epoch,
               "dev_ratio": args.dev_ratio,
               "test_ratio": args.test_ratio,
                "chat":args.chat,
                "seed":args.seed,
                "embedding_dim":args.embedding_dim,
                "patience":args.patience,
                "epsilon":args.epsilon
           }
           )
set_seed(args.seed)
herbs_count = 811
symptoms_count = 390
sh_count = herbs_count + symptoms_count
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
print("lr: ", wandb.config['lr'], " rec: ", wandb.config['rec'], " dropout: ", wandb.config['dropout'], " batch_size: ",
      wandb.config['batch_size'], " epoch: ", wandb.config['epoch'], " dev_ratio: ", wandb.config['dev_ratio'], 
      "test_ratio: ", wandb.config['test_ratio'])


"""创建3种图数据"""
sh_edge = np.load('data/new/SH.npy')  # 2*42419
sh_edge = sh_edge.tolist()
sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)
sh_x = torch.tensor(
    [[i] for i in range(herbs_count+symptoms_count)], dtype=torch.float)  # 节点特征
sh_data = Data(x=sh_x, edge_index=sh_edge_index.contiguous()).to(device)

ss_edge = np.load('data/new/SS.npy')  # 2*5566
ss_edge = ss_edge.tolist()
ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)
ss_x = torch.tensor([[i] for i in range(symptoms_count)], dtype=torch.float)
ss_data = Data(x=ss_x, edge_index=ss_edge_index.contiguous()).to(device)

hh_edge = np.load('data/new/HH.npy')  # 2*65581
hh_edge = hh_edge.tolist()
hh_edge_index = torch.tensor(
    hh_edge, dtype=torch.long) - symptoms_count  # 边索引需要减去390
hh_x = torch.tensor([[i] for i in range(herbs_count)], dtype=torch.float)
hh_data = Data(x=hh_x, edge_index=hh_edge_index.contiguous()).to(device)

# 一共有1201列，前390列是symptoms，后811列是herbs
prescript = load_obj('data/new/prescription_onehot.pkl')
pLen = len(prescript)
print("prescription num: ", pLen)
pS_array = prescript.iloc[:, :symptoms_count].values  # 33765*390
pH_array = prescript.iloc[:, symptoms_count:].values  # 33765*811

kg_oneHot = pickle.load(open('data/new/knowledge_db_hh.pkl', 'rb'))  # 811*811
kg_oneHot = torch.from_numpy(kg_oneHot.values).float().to(device)
chat_symptoms = np.array(pickle.load(
    open('data/new/embeddingswotext.pkl', 'rb')))  # 33765*1536
chat_symptoms = torch.from_numpy(chat_symptoms).float().to(device)
print('chat_symptoms', chat_symptoms.shape)
# 训练集开发集测试集的下标
p_list = list(range(pLen))
x_train, x_dev_test = train_test_split(p_list, test_size=(
    wandb.config['dev_ratio']+wandb.config['test_ratio']), shuffle=False, random_state=wandb.config['seed'])
x_dev, x_test = train_test_split(
    x_dev_test, test_size=1 - 0.5, shuffle=False, random_state=wandb.config['seed'])
print("train_size: ", len(x_train), "dev_size: ",
      len(x_dev), "test_size: ", len(x_test))


train_dataset = presDataset(pS_array[x_train], pH_array[x_train],chat_symptoms[x_train])
dev_dataset = presDataset(pS_array[x_dev], pH_array[x_dev],chat_symptoms[x_dev])
test_dataset = presDataset(pS_array[x_test], pH_array[x_test],chat_symptoms[x_test])


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=wandb.config['batch_size'])
dev_loader = torch.utils.data.DataLoader(
    dev_dataset, batch_size=wandb.config['batch_size'])
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=wandb.config['batch_size'])


model = KDHR(ss_num=symptoms_count, hh_num=herbs_count, sh_num=sh_count,
             embedding_dim=wandb.config['embedding_dim'], batch_size=wandb.config['batch_size'], dropout=wandb.config['dropout'],chat=wandb.config['chat']).to(device)


criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
optimizer = torch.optim.Adam(
    model.parameters(), lr=wandb.config['lr'], weight_decay=wandb.config['rec'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.8)
early_stopping = EarlyStopping(patience=wandb.config['patience'], verbose=True)

wandb.watch(model, criterion, log="all", log_freq=10)

for epoch in range(wandb.config['epoch']):
    model.train()
    running_loss = 0.0
    for i, (sid, hid, chatid) in enumerate(train_loader):
        sid, hid,chatid = sid.float().to(device), hid.float().to(device),chatid.float().to(device)
        optimizer.zero_grad()
        # batch*805 概率矩阵
        outputs = model(sh_data.x, sh_data.edge_index, ss_data.x, ss_data.edge_index,
                        hh_data.x, hh_data.edge_index, sid, kg_oneHot, chatid)
        loss = criterion(outputs, hid)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    wandb.log({'train_loss': running_loss / len(train_loader)})

    model.eval()
    dev_loss = 0
    dev_p5, dev_p10, dev_p20 = 0, 0, 0
    dev_r5, dev_r10, dev_r20 = 0, 0, 0
    dev_f1_5, dev_f1_10, dev_f1_20 = 0, 0, 0
    for tsid, thid, tchatid in dev_loader:
        tsid, thid, tchatid = tsid.float().to(device), thid.float().to(device), tchatid.float().to(device)
        outputs = model(sh_data.x, sh_data.edge_index, ss_data.x, ss_data.edge_index,
                        hh_data.x, hh_data.edge_index, tsid, kg_oneHot, tchatid)
        dev_loss += criterion(outputs, thid).item()
        for i, hid in enumerate(thid):
            trueLabel = []  # 对应存在草药的索引
            for idx, val in enumerate(hid):  # 获得thid中值为一的索引
                if val == 1:
                    trueLabel.append(idx)
            top5 = torch.topk(outputs[i], 5)[1]  # 预测值前5索引
            count = 0
            for m in top5:
                if m in trueLabel:
                    count += 1
            dev_p5 += count / 5
            dev_r5 += count / len(trueLabel)

            top10 = torch.topk(outputs[i], 10)[1]  # 预测值前10索引
            count = 0
            for m in top10:
                if m in trueLabel:
                    count += 1
            dev_p10 += count / 10
            dev_r10 += count / len(trueLabel)

            top20 = torch.topk(outputs[i], 20)[1]  # 预测值前20索引
            count = 0
            for m in top20:
                if m in trueLabel:
                    count += 1
            dev_p20 += count / 20
            dev_r20 += count / len(trueLabel)

    scheduler.step()
    wandb.log({'dev_loss': dev_loss / len(dev_loader)})
    wandb.log({'dev_p_5': dev_p5 / len(x_dev), 'dev_p_10': dev_p10 /
              len(x_dev), 'dev_p_20': dev_p20 / len(x_dev)})
    wandb.log({'dev_r_5': dev_r5 / len(x_dev), 'dev_r_10': dev_r10 /
              len(x_dev), 'dev_r_20': dev_r20 / len(x_dev)})
    wandb.log({'dev_f1_5': 2 * (dev_p5 / len(x_dev)) * (dev_r5 / len(x_dev))/((dev_p5 / len(x_dev))+(dev_r5 / len(x_dev)) + wandb.config['epsilon']),
               'dev_f1_10': 2 * (dev_p10 / len(x_dev)) * (dev_r10 / len(x_dev))/((dev_p10 / len(x_dev))+(dev_r10 / len(x_dev)) + wandb.config['epsilon']),
               'dev_f1_20': 2 * (dev_p20 / len(x_dev)) * (dev_r20 / len(x_dev))/((dev_p20 / len(x_dev))+(dev_r20 / len(x_dev)) + wandb.config['epsilon'])})

    early_stopping(dev_loss / len(dev_loader), model,path='./output/model/checkpoint.pt')
    if early_stopping.early_stop:
        print("Early stopping")
        break


model.load_state_dict(torch.load('checkpoint.pt'))

model.eval()
test_loss = 0
test_p5, test_p10, test_p20 = 0, 0, 0
test_r5, test_r10, test_r20 = 0, 0, 0
test_f1_5, test_f1_10, test_f1_20 = 0, 0, 0

for tsid, thid, tchatid in test_loader:
    tsid, thid, tchatid = tsid.float().to(device), thid.float().to(device), tchatid.float().to(device)
    # 512*805 概率矩阵
    outputs = model(sh_data.x, sh_data.edge_index, ss_data.x,
                    ss_data.edge_index, hh_data.x, hh_data.edge_index, tsid, kg_oneHot, tchatid)
    test_loss += criterion(outputs, thid).item()
    # thid batch*811
    for i, hid in enumerate(thid):
        trueLabel = []  # 对应存在草药的索引
        for idx, val in enumerate(hid):  # 获得thid中值为一的索引
            if val == 1:
                trueLabel.append(idx)

        top5 = torch.topk(outputs[i], 5)[1]  # 预测值前5索引
        count = 0
        for m in top5:
            if m in trueLabel:
                count += 1
        test_p5 += count / 5
        test_r5 += count / len(trueLabel)

        top10 = torch.topk(outputs[i], 10)[1]  # 预测值前10索引
        count = 0
        for m in top10:
            if m in trueLabel:
                count += 1
        test_p10 += count / 10
        test_r10 += count / len(trueLabel)

        top20 = torch.topk(outputs[i], 20)[1]  # 预测值前20索引
        count = 0
        for m in top20:
            if m in trueLabel:
                count += 1
        test_p20 += count / 20
        test_r20 += count / len(trueLabel)
    wandb.log({'test_loss': test_loss / len(test_loader)})
    wandb.log({'test_p_5': test_p5 / len(x_test), 'test_p_10': test_p10 /
               len(x_test), 'test_p_20': test_p20 / len(x_test)})
    wandb.log({'test_r_5': test_r5 / len(x_test), 'test_r_10': test_r10 /
               len(x_test), 'test_r_20': test_r20 / len(x_test)})
    wandb.log({'test_f1_5': 2 * (test_p5 / len(x_test)) * (test_r5 / len(x_test)) / ((test_p5 / len(x_test)) + (test_r5 / len(x_test))),
               'test_f1_10': 2 * (test_p10 / len(x_test)) * (test_r10 / len(x_test)) / ((test_p10 / len(x_test)) + (test_r10 / len(x_test))),
               'test_f1_20': 2 * (test_p20 / len(x_test)) * (test_r20 / len(x_test)) / ((test_p20 / len(x_test)) + (test_r20 / len(x_test)))})
