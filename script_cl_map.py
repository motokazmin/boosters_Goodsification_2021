import pandas as pd
import pickle
import re
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk

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

def tokenize_text(text):
    for word in nltk.word_tokenize(text):
        if word in voc_dict:
            return voc_dict[word]
            
    return 10000


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
voc_dict = vocab_uniq_category_words.to_dict()
res_uniq = pd.DataFrame()
res_uniq['id']   = test_uniq['id']
res_uniq['pred'] = test_uniq.item_name.apply(tokenize_text)

freq_mix = pickle.load(open('freq_mix', 'rb'))
clf = pickle.load(open('clf_task1_mix', 'rb'))

X_test_mix = freq_mix.transform(test_mix.item_name)
onehot = Binarizer()
X_test_mix = onehot.fit_transform(X_test_mix)

pred_mix = clf.predict(X_test_mix)

res_mix = pd.DataFrame({'id' : test_mix['id'].values, 'pred' : pred_mix})
res_uniq.append(res_mix).sort_values('id').to_csv('answers.csv', index=None)

