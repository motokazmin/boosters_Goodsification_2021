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
    
    result = ''
    for word in nltk.word_tokenize(name.lower()):
        if word == 'nan':
            word = 'krvnan'

        result += ' ' + word

    return result.strip()

test = pd.read_parquet('data/task1_test_for_user.parquet')
test['item_name'] = test['item_name'].apply(clean_name)

vocab_uniq_category_words = pd.read_csv('uniq_words.csv', index_col = 0)['category_id']
freq_on_uniq_words = CountVectorizer(binary=True)
freq_on_uniq_words.fit(vocab_uniq_category_words.index)
test_transformed = pd.DataFrame(freq_on_uniq_words.transform(test.item_name).toarray(),
                                   columns = freq_on_uniq_words.get_feature_names(), index = test.index)
test_on_uniq_indexes = test_transformed[(test_transformed.sum(axis = 1) > 0).values].index

test_mix  = test[test.index.isin(test_on_uniq_indexes) == False]
test_uniq = test[test.index.isin(test_on_uniq_indexes) == True]

freq_mix = pickle.load(open('freq_mix', 'rb'))
freq_uniq = pickle.load(open('freq_uniq', 'rb'))
clf_task1_mix = pickle.load(open('clf_task1_mix', 'rb'))
clf_task1_uniq = pickle.load(open('clf_task1_uniq', 'rb'))

X_test_mix = freq_mix.transform(test_mix.item_name)
onehot = Binarizer()
X_test_mix = onehot.fit_transform(X_test_mix)

pred_mix = clf_task1_mix.predict(X_test_mix)

res_mix = pd.DataFrame({'id' : test_mix['id'].values, 'pred' : pred_mix})

X_test_uniq = freq_uniq.transform(test_uniq.item_name)
onehot = Binarizer()
X_test_uniq = onehot.fit_transform(X_test_uniq)

pred_uniq = clf_task1_uniq.predict(X_test_uniq)

res_uniq = pd.DataFrame({'id' : test_uniq['id'].values, 'pred' : pred_uniq})

res_uniq.append(res_mix).to_csv('answers.csv', index=None)

