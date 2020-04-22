from tqdm import tqdm
from sklearn.model_selection import KFold
from gru_pooled import *


label_map = {'-1': 0, '0': 1, '1': 2}
index_map = {v: k for k, v in label_map.items()}

def gen_res2(model, tokenizer):
    model = model.to("cuda:"+str(DEV_NUM))
    df = pd.read_csv("data/nCov_10k_test.csv").fillna("")
    testset = SentimentDataset("test", tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    label_map = {'-1': 0, '0': 1, '1': 2}

    print("We have " + str(len(testset)) + " to generate...")

    # 用分類模型預測測試集
    predictions = get_predictions(model, testloader)
    print("We have " + str(len(predictions)) + " generated...")
    # 用來將預測的 label id 轉回 label 文字
    return predictions.tolist()


tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
#training
df = pd.read_csv("data/nCoV_100k_train.labled.csv")
df = df.sample(frac=SAMPLE_FRAC, random_state=9527)
# 去除不必要的欄位並重新命名兩標題的欄位名
df = df.reset_index()
if args.small:
    df = df.loc[:2000, ['微博中文内容', '情感倾向']]
else:
    df = df.loc[:, ['微博中文内容', '情感倾向']]
df.columns = ['text_a', 'label']
df['text_a'] = df.text_a.apply(remove_puncs)
bad_rows = ((df['label'] != '-1') & (df['label'] != '0') & (df['label'] != '1'))
df = df[~bad_rows]


predictions = []
kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df)):
    train_fold = df.iloc[trn_idx]
    valid_fold = df.iloc[val_idx]
    print(get_time() + ": 訓練樣本數：", len(train_fold))
    trainset = SentimentDataset("train", tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, df_=train_fold)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)
    validset = SentimentDataset("valid", tokenizer=tokenizer,max_seq_length=MAX_SEQ_LENGTH, df_=valid_fold)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE)

    model = Bert_GRU_Pooled.from_pretrained(PRETRAINED_MODEL_PATH, num_labels=NUM_LABELS)
    train_model(model, trainloader, validloader, 'kfold_tmp_gru_' + ("large" if args.zhlarge else "small"))
    del model
    torch.cuda.empty_cache()

    model = Bert_GRU_Pooled.from_pretrained(PRETRAINED_MODEL_PATH, num_labels=NUM_LABELS)
    model.load_state_dict(torch.load('model_output/' + 'kfold_tmp_gru_' + ("large" if args.zhlarge else "small")))
    pred = gen_res2(model, tokenizer)
    predictions.append(pred)
    del model
    torch.cuda.empty_cache()

predictions = np.array(predictions)
voted_predictions = []
for i in range(predictions.shape[1]):
    cur_label_list = []
    for j in range(predictions.shape[0]):
        cur_label_list.append(predictions[j][i])
    maxlabel = max(cur_label_list, key=cur_label_list.count)
    voted_predictions.append(maxlabel)

df = pd.read_csv("data/nCov_10k_test.csv").fillna("")
df_out = pd.DataFrame({"y": voted_predictions})
df_out['y'] = df_out.y.apply(lambda x: index_map[x])
df_pred = pd.concat([df.loc[:, ["微博id"]], 
                            df_out.loc[:, 'y']], axis=1)
df_pred.to_csv('kfold_gru_' + ("large" if args.zhlarge else "small") + '-final.csv', index=None)
