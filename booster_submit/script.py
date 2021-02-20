import pandas as pd
import pickle
import re
import nltk
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer

def clean_name(name):
    if pd.isna(name):
        return ''
    
    name = re.sub(r'[^\w]', ' ', name)
    name = re.sub(r'[,:._\-\[\]\d]', ' ', name)
    return name.lower().strip()

test = pd.read_parquet('data/task1_test_for_user.parquet')
test['item_name'] = test['item_name'].apply(clean_name)

freq_bi = pickle.load(open('freq_bi', 'rb'))
freq_false = pickle.load(open('freq_false', 'rb'))
freq_true = pickle.load(open('freq_true', 'rb'))

clf_bi = pickle.load(open('clf_task1_bi', 'rb'))
clf_false = pickle.load(open('clf_task1_false', 'rb'))
clf_true = pickle.load(open('clf_task1_true', 'rb'))

onehot = Binarizer()
X_test_bi = freq_bi.transform(test.item_name)
X_test_bi = onehot.fit_transform(X_test_bi)
pred_bi = clf_bi.predict(X_test_bi)

test_true = test[pred_bi  == True]
test_false = test[pred_bi == False]

X_test_false = freq_false.transform(test_false.item_name)
X_test_false = onehot.fit_transform(X_test_false)
pred_false = clf_false.predict(X_test_false)

X_test_true = freq_true.transform(test_true.item_name)
X_test_true = onehot.fit_transform(X_test_true)
pred_true = clf_true.predict(X_test_true)

res_false = pd.DataFrame({'id' : test_false['id'].values, 'pred' : pred_false})
res_true  = pd.DataFrame({'id' : test_true['id'].values, 'pred' : pred_true})

res_false.append(res_true).to_csv('answers.csv', index=None)

