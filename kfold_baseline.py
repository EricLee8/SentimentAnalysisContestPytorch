import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from simpletransformers.classification import ClassificationModel

label_map = {'-1': 0, '0': 1, '1': 2}
index_map = {v: k for k, v in label_map.items()}

train = pd.read_csv('train.tsv', sep="\t").fillna("完全中立")
test = pd.read_csv('data/nCov_10k_test.csv').fillna("完全中立")
test = test.loc[:, ['微博中文内容']]
test.columns = ['text_a']
test[['text_a']] = test[['text_a']].astype(str)

train.columns = ['text_a', 'labels']
train[['text_a']] = train[['text_a']].astype(str)
train['labels'] = train.labels.apply(lambda x: label_map[str(x)])
# train = train.loc[:500, :]
print(len(train))

train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 2,
}

predictions = []

kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train)):
    train_fold = train.iloc[trn_idx]
    valid_fold = train.iloc[val_idx]
    
    model = ClassificationModel('bert', 
                            'chinese', 
                            num_labels=3, 
                            use_cuda=True, 
                            cuda_device=3, 
                            args=train_args)
    
    model.train_model(train_fold, eval_df=valid_fold)
    result, model_outputs, wrong_predictions = model.eval_model(eval_df=valid_fold)
    print("===========================================================================================")
    print(result)
    print("===========================================================================================")
    text_list= list()
    for i, row in tqdm(test.iterrows()):
        text_list.append(row['text_a'])
    pred, _ = model.predict(text_list)
    predictions.append(pred)

# print(predictions)
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
df_pred.to_csv('final.csv', index=None)
