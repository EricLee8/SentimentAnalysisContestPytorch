import torch
from transformers import BertTokenizer, BertForPreTraining
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import jieba
import time


PRETRAINED_MODEL_NAME = "chinese/"
DEV_NUM = 1
BATCH_SIZE = 32
EPOCHS = 2
MAX_SEQ_LENGTH = 128
INTERVAL = 50


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))


def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    masked_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    
    return inputs, labels


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


class PreTrainingDataset(Dataset):
    def __init__(self, tokenizer, max_seq_length):
        self.df = pd.read_csv("preTrain" + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
    def __getitem__(self, idx):
        '''
        returns: (tensors) 
        input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label
        '''
        text_a = self.df.iloc[idx].text[:self.max_seq_length-3]
        if random.randint(0, 1) == 0: # text_b is sampled randomly from the corpus, irrelevant to text_a
            L = list(jieba.cut(text_a))
            if len(L) >= 2:
                text_a = "".join(L[: len(L)//2])[:self.max_seq_length//2 - 2]
            L_b = list(jieba.cut(self.df.iloc[random.randint(0, self.len-1)].text[:self.max_seq_length]))
            text_b = "".join(L_b[len(L_b)//2 :])[:self.max_seq_length//2 - 2] if len(L_b) >= 2 else "".join(L_b)
            next_sentence_label = torch.tensor([0])
        else: # text_b is relevant to text_a
            L = list(jieba.cut(text_a))
            if len(L) >= 2:
                text_a = "".join(L[: len(L)//2])
                text_b = "".join(L[len(L)//2 :])
                next_sentence_label = torch.tensor([1])
            else: # a不够长，还是把a、b设置为不相关
                L_b = list(jieba.cut(self.df.iloc[random.randint(0, self.len-1)].text[:self.max_seq_length]))
                text_b = "".join(L_b[len(L_b)//2 :])[:self.max_seq_length//2 - 2] if len(L_b) >= 2 else "".join(L_b)
                next_sentence_label = torch.tensor([0])
        
        if len(text_a) <= 1 or len(text_b) <= 1:
            text_a = '茕茕孑立'
            text_b = '沆瀣一气'
            next_sentence_label = torch.tensor([1])
        
        texta_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text_a))
        textb_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text_b))
        _truncate_seq_pair(texta_ids, textb_ids, self.max_seq_length-3)
        texta_ids = torch.tensor(texta_ids, dtype=torch.long)
        textb_ids = torch.tensor(textb_ids, dtype=torch.long)

        try:
            labels_a = mask_tokens(texta_ids, self.tokenizer)[1]
            labels_b = mask_tokens(textb_ids, self.tokenizer)[1]
        except BaseException as e:
            print(e)
            print(text_a)
            print(text_b)

        masked_lm_labels = torch.cat((torch.tensor([101]), labels_a, torch.tensor([102]), labels_b, torch.tensor([102])))
        cur_len = masked_lm_labels.shape[0]
        if cur_len < self.max_seq_length:
            masked_lm_labels = torch.cat((masked_lm_labels, torch.tensor([-100]*(self.max_seq_length-cur_len))))
        
        d = self.tokenizer.encode_plus(text_a, text_b, max_length=128, pad_to_max_length=True)
        input_ids = torch.tensor(d['input_ids'])
        token_type_ids = torch.tensor(d['token_type_ids'])
        attention_mask = torch.tensor(d['attention_mask'])
        
        assert len(input_ids) == self.max_seq_length
        assert len(token_type_ids) == self.max_seq_length
        assert len(attention_mask) == self.max_seq_length
        assert len(masked_lm_labels) == self.max_seq_length
        return input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label
        
    def __len__(self):
        return self.len


def train_model(model, trainloader, length):
    print("============================================= Start training =============================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    # 使用 Adam Optim 更新整個分類模型的參數
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        iter_num = 0
        for data in trainloader:
            input_ids, token_type_ids, attention_mask, masked_lm_labels,\
                next_sentence_label = [t.to(device) for t in data]
            iter_num += 1

            optimizer.zero_grad()
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                masked_lm_labels = masked_lm_labels,
                next_sentence_label = next_sentence_label
            )

            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if iter_num % INTERVAL == 0:
                print(get_time() + (': [epoch %d], [iter %d/%d] average loss: %.3f' % (epoch + 1, iter_num, length, running_loss/INTERVAL)) )
                running_loss = 0.0


if __name__ == '__main__':
    print(get_time() + ": Start loading data...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForPreTraining.from_pretrained(PRETRAINED_MODEL_NAME)
    # model = torch.nn.DataParallel(model)
    trainset = PreTrainingDataset(tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    length = len(trainset) // BATCH_SIZE
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)
    print(get_time() + ": Data has already been loaded...")
    train_model(model, trainloader, length)
    model.save_pretrained('preTrained/')
    print(get_time() + ": Pretraining process done...")