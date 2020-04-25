import pandas as pd

puncs = ['?', '展开全文c']
def remove_puncs(text: str):
    if not isinstance(text, str):
        return ''
    for ele in puncs:
        text = text.replace(ele, '')
    return text

df = pd.read_csv("data/nCoV_900k_train.unlabled.csv")
df = df.reset_index()
df = df.loc[:, ['微博中文内容']]
df.rename(columns={'微博中文内容': 'text'}, inplace=True)
print(len(df))
bad_rows = df['text'].isnull()
df = df[~bad_rows]
print(len(df))
bad_rows = df.text.apply(lambda x: len(x))<10
df = df[~bad_rows]
print(len(df))
df['text'] = df.text.apply(remove_puncs)
df.to_csv("preTrain.tsv", sep="\t", index=False)