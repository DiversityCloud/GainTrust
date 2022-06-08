import h5py
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import pickle
import time
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn import metrics
from CreateSet import my_collate, MyDataset, get_user_cate, get_user_data, get_sim_user_data
from SelfAttention import BertAttention, SelfAttentionConfig


##实验配置##
validation_split = 0.5 #测试集划分比例
LSTM_LAYER = 1 #LSTM层数
N_LINEAR_LAYERS = 1
DATA_SET = "1"
MLP_LAYERS = 2
MLP_USE_RELU = True

EMBEDDING_SIZE = 64
HIDDEN_SIZE = 368
FEATURE_CLASS = 64

def train():
    print(f"开始训练---DATA_SET: {DATA_SET}, SPLIT: {validation_split}")
    save_dir = f"./save/data{DATA_SET}_trainratio{1-validation_split}_attentionhead4_lstmlayer{LSTM_LAYER}_emb{EMBEDDING_SIZE}_linear{N_LINEAR_LAYERS}_MLP{MLP_LAYERS}_{'RELU' if MLP_USE_RELU else 'NORELU'}" # 模型保存路径，用时间戳，每次训练保存一份新的
    os.makedirs(save_dir, exist_ok=True)

    if DATA_SET == "1":
        data_set_dir = "./data/first_data"
        data_all = np.loadtxt('./data/first_data/All_stamp_Sameuser_item.txt')
        label_data = np.loadtxt('./data/first_data/epinion_trust_with_timestamp_new.txt')
    elif DATA_SET == "2":
        data_set_dir = "./data/second_data"
        data_all = np.loadtxt('./data/second_data/Ciao_rating_stamp.txt')
        label_data = np.loadtxt('./data/second_data/Ciao_trust_new.txt')
    else:
        raise Exception("DATA_SET ERROR")

    class DoubleLstmNet(nn.Module):
        def __init__(self):
            super(DoubleLstmNet, self).__init__()
            self.embeddings0 = nn.Embedding(300000, EMBEDDING_SIZE)
            self.embeddings1 = nn.Embedding(10, EMBEDDING_SIZE)
            self.embeddings2 = nn.Embedding(10, EMBEDDING_SIZE)
            self.embeddings3 = nn.Embedding(30, EMBEDDING_SIZE)
            self.embeddings_usr = nn.Embedding(30000, EMBEDDING_SIZE)

            # up layer lstm net + linear layer
            self.rnn_up = nn.LSTM(
                input_size=1472*EMBEDDING_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=LSTM_LAYER,
                batch_first=False,  # (time_step,batch, input_size)
            )
            self.out_up = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for i in range(N_LINEAR_LAYERS-1)])
            self.out_up.append(torch.nn.Linear(HIDDEN_SIZE, FEATURE_CLASS))
            # down layer lstm net + linear layer
            self.rnn_down = nn.LSTM(
                input_size=1472*EMBEDDING_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=LSTM_LAYER,
                batch_first=False,  # (time_step, batch, input_size)
            )
            # down layer lstm net + linear layer
            """这里和上一个线性层都有sigmoid激活函数，可以看一下在sigmoid的输入过大或过小饱和特点会不会使得特征消失或改变"""
            self.out_down = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for i in range(N_LINEAR_LAYERS-1)])
            self.out_down.append(torch.nn.Linear(HIDDEN_SIZE, FEATURE_CLASS))

            # self.feature_size = FEATURE_CLASS*2+EMBEDDING_SIZE*2
            # self.self_attention_feature_linear_1 = nn.Linear(FEATURE_CLASS, 64)
            # self.self_attention_feature_linear_2 = nn.Linear(64, 64*32)

            self.linear_for_lstm = nn.Linear(FEATURE_CLASS, 64)
            self.linear_for_emb = nn.Linear(EMBEDDING_SIZE, 64)

            attn_config = SelfAttentionConfig()
            attn_config.hidden_size = 64
            self.self_attention = BertAttention(attn_config)

            # 最后分类 MLP 层
            self.out_linear_list = nn.ModuleList([])
            feature_size = 64*4
            for _ in range(MLP_LAYERS-1):
                next_feature_size = feature_size // 2
                self.out_linear_list.append(torch.nn.Linear(feature_size, next_feature_size))
                feature_size = next_feature_size
            self.out_linear_list.append(torch.nn.Linear(feature_size, 2))

        def forward(self, input1, input2, id1, id2, score1, score2):
            # get sim usr embedding
            id1 = id1.to(torch.long)
            id2 = id2.to(torch.long)
            id1_emb = self.embeddings_usr(id1)
            id2_emb = self.embeddings_usr(id2)
            score_p1 = score1 / torch.sum(score1, dim=1, keepdim=True)
            score_p2 = score2 / torch.sum(score2, dim=1, keepdim=True)
            id1_emb_weight = id1_emb * (score_p1.unsqueeze(2))
            id1_emb_weight = torch.sum(id1_emb_weight, dim=1)
            id2_emb_weight = id2_emb * (score_p2.unsqueeze(2))
            id2_emb_weight = torch.sum(id2_emb_weight, dim=1)

            # output of up lstm
            input1 = input1.to(torch.long)
            input2 = input2.to(torch.long)

            lemb0 = self.embeddings0(input1[:,:,0:368])
            lemb1 = self.embeddings1(input1[:,:,368:736])
            lemb2 = self.embeddings2(input1[:,:,736:1104])
            lemb3 = self.embeddings3(input1[:,:,1104:1472])
            lemb = torch.cat([lemb0,lemb1,lemb2,lemb3],dim=2)
            t, b, l, f = lemb.shape
            lemb = lemb.view(t,b,l*f)

            remb0 = self.embeddings0(input2[:,:,0:368])
            remb1 = self.embeddings1(input2[:,:,368:736])
            remb2 = self.embeddings2(input2[:,:,736:1104])
            remb3 = self.embeddings3(input2[:,:,1104:1472])
            remb = torch.cat([remb0,remb1,remb2,remb3],dim=2)
            remb = remb.view(t,b,l*f)

            r_out_up, (h_n_u, h_c_u) = self.rnn_up(lemb)
            r_out_up = F.dropout(r_out_up, p=0.4)
            out_up = r_out_up.mean(dim=0)
            for l in self.out_up:
                out_up = l(out_up)  # shape(batch_size, time_step, input_size, embedding_size)

            r_out_down, (h_n_d, h_c_d) = self.rnn_down(remb)
            r_out_down = F.dropout(r_out_down, p=0.4)
            out_down = r_out_down.mean(dim=0)
            for l in self.out_down:
                out_down = l(out_down)  # shape(batch_size, time_step, input_size, embedding_size)


            out_up = self.linear_for_lstm(out_up.to(torch.float))
            id1_emb_weight = self.linear_for_emb(id1_emb_weight.to(torch.float))
            out_down = self.linear_for_lstm(out_down.to(torch.float))
            id2_emb_weight = self.linear_for_emb(id2_emb_weight.to(torch.float))
            concat_self_attn_emb = torch.stack([id1_emb_weight, out_up, id2_emb_weight, out_down], dim=1)

            # concat_emb = torch.cat([id1_emb_weight, out_up, id2_emb_weight, out_down], dim=1)
            # concat_emb = concat_emb.to(torch.float)
            # concat_self_attn_emb = self.self_attention_feature_linear_1(concat_emb)
            # concat_self_attn_emb = self.self_attention_feature_linear_2(concat_self_attn_emb)
            # concat_self_attn_emb = torch.reshape(concat_self_attn_emb, shape=[concat_self_attn_emb.shape[0], 64, -1])

            logits, multi_head_attention_scores = self.self_attention(concat_self_attn_emb)
            logits = torch.reshape(logits, shape=[logits.shape[0], -1])
            for out_linear in self.out_linear_list[:-1]:
                logits = F.relu(out_linear(logits))
            logits = self.out_linear_list[-1](logits)
            return logits, multi_head_attention_scores

    DoubleNet = DoubleLstmNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DoubleNet.to(device)
    optimizer = torch.optim.Adam(DoubleNet.parameters(), lr=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()

    user_cate = get_user_cate(data_all)
    # 判断是否已经加载为 pkl 文件
    if os.path.exists(os.path.join(data_set_dir, "all_users_data_dict.pkl")):
        print("load data from pkl...")
        all_user_data_dict = pickle.load(open(os.path.join(data_set_dir, "all_users_data_dict.pkl"), "rb"))
    else:
        print("load data from get_user_data...")
        all_user_data_dict = get_user_data(user_cate, data_all)
    all_users_sim_dict = get_sim_user_data(label_data)
    print('dict okay!')
    dataset = MyDataset(all_user_data_dict, all_users_sim_dict, user_cate, label_data)

    b0, b1, id0, id1, score0, score1, label = dataset[0]
    print(b0.shape, b1.shape, label)
    print(id0.shape, id1.shape, label)
    print(score0.shape, score1.shape, label)

    BatchSize = 32
    shuffle_dataset = True
    random_seed= 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    trainloader = DataLoader(dataset=dataset,
            batch_size=BatchSize,
            collate_fn=my_collate,
            pin_memory=True,
            sampler=train_sampler
            )

    testloader = DataLoader(dataset=dataset,
            batch_size=BatchSize,
            collate_fn=my_collate,
            pin_memory=True,
            sampler=valid_sampler
            )


    def predict(score,fz):
        x = torch.zeros_like(score)
        y = torch.ones_like(score)
        return torch.where(score>fz, y, x)

    last_acc = 1
    flag = 0

    for epoch in range(20):
        correct = 0
        total_loss = 0
        tp = 0
        tp_fp = 0
        tp_fn = 0
        DoubleNet.train()
        for ix, batch in enumerate(trainloader):
            input1, input2, id0, id1, score0, score1, label = batch
            input1 = input1.to(device)
            input2 = input2.to(device)
            id0 = id0.to(device)
            id1 = id1.to(device)
            score0 = score0.to(device)
            score1 = score1.to(device)
            label = torch.tensor(label,dtype=torch.long)
            label = label.to(device)
            optimizer.zero_grad()
            output, _ = DoubleNet(input1, input2, id0, id1, score0, score1)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print('epoch',epoch,'batch',ix+1,'loss:',total_loss/(ix+1))

        torch.save(DoubleNet.state_dict(), f"{save_dir}/model_{epoch}.pth")

        correct = torch.zeros(11)
        tp = torch.zeros(11)
        tp_fp = torch.zeros(11)
        tp_fn = torch.zeros(11)
        fp = torch.zeros(11)
        fp_tn = torch.zeros(11)
        multi_head_attention_matrix_all = None
        save_attention_matrix_nums = 0
        total_loss = 0
        DoubleNet.eval()
        print('start validation <<<<<<<<<<<< <<<<<<<<<')
        for ix,batch in enumerate(testloader):
            input1, input2, id0, id1, score0, score1, label = batch
            input1 = input1.to(device)
            input2 = input2.to(device)
            id0 = id0.to(device)
            id1 = id1.to(device)
            score0 = score0.to(device)
            score1 = score1.to(device)
            label = torch.tensor(label,dtype=torch.long)
            label = label.to(device)
            output, attention_matrix = DoubleNet(input1, input2, id0, id1, score0, score1)
            attention_matrix = attention_matrix.cpu().detach().numpy()
            if multi_head_attention_matrix_all is None:
                multi_head_attention_matrix_all = attention_matrix
                save_attention_matrix_nums += 1
            elif save_attention_matrix_nums < 1000:
                multi_head_attention_matrix_all = np.vstack([multi_head_attention_matrix_all, attention_matrix])
                save_attention_matrix_nums += 1
            loss = loss_func(output, label)
            total_loss += loss.item()
            output = torch.softmax(output, dim=1)
            output = output[:, 1]
            for i,fz in enumerate([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]):
                pred = predict(output,fz)
                correct[i] += (pred == label).sum().item()
                tp[i] += pred.logical_and(label).sum().item() # 预测1实际1
                tp_fp[i] += pred.sum().item()
                tp_fn[i] += label.sum().item()
                fp[i] += pred.logical_and(label.logical_not()).sum().item() # 预测1实际0
                fp_tn[i] += label.logical_not().sum().item()
        acc = correct/(BatchSize*(ix+1))
        print(acc)
        precision = tp/tp_fp
        tpr = recall = tp/tp_fn
        f1 = 2*precision*recall/(precision+recall)
        fpr = fp/fp_tn
        auc = metrics.auc(fpr.numpy(), tpr.numpy())
        valid_loss = total_loss/(ix+1)

        with open(f"{save_dir}/res_%s_%s_%s_{epoch}.txt"%(str(1-validation_split),LSTM_LAYER,N_LINEAR_LAYERS), "w", encoding="utf-8") as f:
            f.writelines("valid_loss:\t%0.4f\n" % valid_loss)
            f.writelines("auc:\t%0.4f\n" % auc)

            acc_str = ["%0.4f"% a for a in acc]
            f.writelines("acc:\t"+"\t".join(acc_str)+"\n")

            precision_str = ["%0.4f"% a for a in precision]
            f.writelines("precision:\t"+"\t".join(precision_str)+"\n")

            recall_str = ["%0.4f"% a for a in recall]
            f.writelines("recall:\t"+"\t".join(recall_str)+"\n")

            f1_str = ["%0.4f"% a for a in f1]
            f.writelines("f1:\t"+"\t".join(f1_str)+"\n")

            tpr_str = ["%0.4f"% a for a in tpr]
            f.writelines("tpr:\t"+"\t".join(tpr_str)+"\n")

            fpr_str = ["%0.4f"% a for a in fpr]
            f.writelines("fpr:\t"+"\t".join(fpr_str)+"\n")

        np.save(f"{save_dir}/attention_score_{epoch}.npy", multi_head_attention_matrix_all)

        if abs(acc[-1].numpy() - last_acc) < 0.00001:
            flag += 1
            if flag > 20:
                print("Early stopping at:", epoch)
                break
        else:
            last_acc = acc[-1].numpy()

if __name__ == "__main__":
    for data_set in ["1", "2"]:
        for lstm_layer in [5,4,3,2,1]:
            for emb_size in [128,96,64,32,16]:
                EMBEDDING_SIZE = emb_size
                LSTM_LAYER = lstm_layer
                DATA_SET = data_set
                train()
