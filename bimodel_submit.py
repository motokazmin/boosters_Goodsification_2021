import pandas as pd
import pickle
import re
from sklearn.preprocessing import Binarizer

def clean_name(name):
    name = re.sub(r'[^\w]', ' ', name)
    name = re.sub(r'[,:._\-\[\]\d]', ' ', name)
    return name.lower().strip()

def model_predict(test, tfd_clf, cnv_clf, tfd_freq, cnv_freq):
    p1 = tfd_clf._predict_proba_lr(tfd_freq.transform(test.item_name))
    p2 = cnv_clf._predict_proba_lr(cnv_freq.transform(test.item_name))

    r = []
    for row in range(0, len(p1)):
        mp1 = max(p1[row])
        mp2 = max(p2[row])

        idx_mp1 = np.where(p1[row] == mp1)
        idx_mp2 = np.where(p2[row] == mp2)

        if mp1 > mp2:
            r.extend(tfd_clf.classes_[idx_mp1])
        else:
            r.extend(cnv_clf.classes_[idx_mp2])
    return r


test = pd.read_parquet('data/task1_test_for_user.parquet')
test['item_name'] = test['item_name'].apply(clean_name)

tfd_clf = pickle.load(open('tfd_clf', 'rb'))
cnv_clf = pickle.load(open('cnv_clf', 'rb'))
tfd_freq = pickle.load(open('tfd_freq', 'rb'))
cnv_freq = pickle.load(open('cnv_freq', 'rb'))


pred = model_predict(test, tfd_clf, cnv_clf, tfd_freq, cnv_freq)

res = pd.DataFrame(pred, columns=['pred'])
res['id'] = test['id']

res[['id', 'pred']].to_csv('answers.csv', index=None)

