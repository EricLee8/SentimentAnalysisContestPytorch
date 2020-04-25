import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertPreTrainedModel
import random
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
import time


PRETRAINED_MODEL_NAME = "bert-base-chinese"
BATCH_SIZE = 64
SAMPLE_FRAC = 1
NUM_LABELS = 3
EPOCHS = 3
MAX_SEQ_LENGTH = 128
TRAIN_RATE = 0.9
DEV_NUM = 1


class Bert_Plus_TextCNN(BertPreTrainedModel):
    def __init__(self, config, filter_sizes=[1,2,3,4], num_filters=192):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.convs = nn.ModuleList([
            nn.Conv2d(1, out_channels=num_filters, kernel_size=(fs, config.hidden_size), stride=(1, 1)) for fs in filter_sizes
        ])
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(len(filter_sizes)*num_filters, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
        head_mask=None, inputs_embeds=None, labels=None):
    
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)

        embedded = outputs[0] # (batch_size, sequence_length, hidden_size)
        embedded = embedded.unsqueeze(1) # (batch_size, 1, sequence_length, hidden_size), 1 is channels
        conveds = [F.relu(conv(embedded).squeeze(3)) for conv in self.convs] # (batch_size, num_filters, seq_len-fs+1) for each element
        pooled = [F.max_pool1d(conved, conved.shape[2]).squeeze(2) for conved in conveds] # pooled: (batch_size, num_filters)
        catted = self.dropout(torch.cat(pooled, dim=1)) # (batch_size, len(filter_sizes)*num_filters), default (B, 768(192*4))
        logits = self.classifier(catted)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))) + ": "


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class SentimentDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, tokenizer, max_seq_length):
        assert mode in ["train", "test", "valid"]  # 一般訓練你會需要 dev set
        self.mode = mode
        # 大數據你會需要用 iterator=True
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.is_b = True if self.df.iloc[0].shape[0] == 3 else False
        self.len = len(self.df)
        self.label_map = {'-1': 0, '0': 1, '1': 2}
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if self.mode == "test":
            if self.is_b:
                text_a, text_b = self.df.iloc[idx, :2].values
            else:
                text_a, text_b = self.df.iloc[idx][0], None
            label_tensor = torch.tensor([0])
        else:
            if self.is_b:
                text_a, text_b, label = self.df.iloc[idx, :].values
            else:
                text_a, label = self.df.iloc[idx, :].values
                text_b = None
            label_id = self.label_map[str(label)]
            label_tensor = torch.tensor(label_id)
            
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        if self.is_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length-3) # for [CLS],[SEP],[SEP]
            word_pieces += tokens_a+["[SEP]"]+tokens_b+["[SEP]"]
            segments_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)
        else:
            if len(tokens_a) > self.max_seq_length-2: # for [CLS] and [SEP]
                tokens_a = tokens_a[:self.max_seq_length-2]
            word_pieces += tokens_a+["[SEP]"]
            segments_ids = [0]*(len(tokens_a)+2)
        input_ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        input_mask = [1]*len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segments_ids.append(0)
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segments_ids) == self.max_seq_length

        # 转为tensor
        tokens_tensor = torch.tensor(input_ids)
        segments_tensor = torch.tensor(segments_ids, dtype=torch.long)
        masks_tensor = torch.tensor(input_mask, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, masks_tensor, label_tensor)
    
    def __len__(self):
        return self.len


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
    print(get_time() + "Starting pred/valid...")
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:"+str(DEV_NUM)) for t in data if t is not None]
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    print(get_time() + "Endding pred/valid...")
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions


def gen_res(model, tokenizer):
    df = pd.read_csv("data/nCov_10k_test.csv")
    df2 = df.reset_index()
    df2 = df2.loc[:, ['微博中文内容']]
    df2.columns = ['text_a']
    df2.to_csv("test.tsv", sep="\t", index=False)

    testset = SentimentDataset("test", tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    label_map = {'-1': 0, '0': 1, '1': 2}

    # 用分類模型預測測試集
    predictions = get_predictions(model, testloader)
    # 用來將預測的 label id 轉回 label 文字
    index_map = {v: k for k, v in testset.label_map.items()}
    # 生成 Kaggle 繳交檔案
    df_out = pd.DataFrame({"y": predictions.tolist()})
    df_out['y'] = df_out.y.apply(lambda x: index_map[x])
    df_pred = pd.concat([df.loc[:, ["微博id"]], 
                            df_out.loc[:, 'y']], axis=1)
    df_pred.to_csv('Json-final.csv', index=False)


# 訓練模式
def train_model(model, trainloader, validloader):
    device = torch.device("cuda:"+str(DEV_NUM) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    # 使用 Adam Optim 更新整個分類模型的參數
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            tokens_tensors, segments_tensors, \
            masks_tensors, labels = [t.to(device) for t in data]
            # 將參數梯度歸零
            optimizer.zero_grad()
            # forward pass
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)
            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()
            # 紀錄當前 batch loss
            running_loss += loss.item()

        # 計算分類準確率
        _, acc = get_predictions(model, trainloader, compute_acc=True)
        print('[epoch %d] loss: %.3f, acc: %.3f' %
              (epoch + 1, running_loss, acc))
        _, acc = get_predictions(model, validloader, compute_acc=True)
        print('[epoch %d] valid_loss: %.3f, valid_acc: %.3f' %
              (epoch + 1, running_loss, acc))

def main():
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    df = pd.read_csv("data/nCoV_100k_train.labled.csv")
    df = df.sample(frac=SAMPLE_FRAC, random_state=9527)
    # 去除不必要的欄位並重新命名兩標題的欄位名
    df = df.reset_index()
    df = df.loc[:, ['微博中文内容', '情感倾向']]
    df.columns = ['text_a', 'label']
    bad_rows = ((df['label'] != '-1') & (df['label'] != '0') & (df['label'] != '1'))
    df = df[~bad_rows]
    df_train = df.sample(frac=TRAIN_RATE, random_state=114514)
    df_valid = df[~df.index.isin(df_train.index)]
    df_train.to_csv("train.tsv", sep="\t", index=False)
    df_valid.to_csv("valid.tsv", sep="\t", index=False)
    print("訓練樣本數：", len(df_train))
    trainset = SentimentDataset("train", tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)
    validset = SentimentDataset("valid", tokenizer=tokenizer,max_seq_length=MAX_SEQ_LENGTH)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE)
    model = Bert_Plus_TextCNN.from_pretrained("chinese/", num_labels=NUM_LABELS)
    train_model(model, trainloader, validloader)
    gen_res(model, tokenizer)


if __name__ == '__main__':
    main()