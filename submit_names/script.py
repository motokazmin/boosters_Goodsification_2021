import pandas as pd
import pickle
import re
from sklearn.preprocessing import Binarizer

def clean_name(name):
    name = re.sub(r'[^\w]', ' ', name)
    name = re.sub(r'[,:._\-\[\]\d]', ' ', name)
    return name.lower().strip()


test = pd.read_parquet('data/task1_test_for_user.parquet')
test['item_name'] = test['item_name'].apply(clean_name)

tfidf = pickle.load(open('tfidf', 'rb'))
clf = pickle.load(open('clf_task1', 'rb'))

X_test = tfidf.transform(test.item_name)
onehot = Binarizer()
X_test = onehot.fit_transform(X_test)

pred = clf.predict(X_test)

res = pd.DataFrame(pred, columns=['pred'])
res['id'] = test['id']

res[['id', 'pred']].to_csv('answers.csv', index=None)
