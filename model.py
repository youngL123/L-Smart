from data_process import data_process
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

clean_data, labels, data_after_stop = data_process()

data_tr, data_te, labels_tr, labels_te = train_test_split(clean_data, labels, test_size=0.2)

# 文本的向量化表示
count_vectorizer = CountVectorizer()
data_tr = count_vectorizer.fit_transform(data_tr)
X_tr = TfidfTransformer().fit_transform(data_tr.toarray()).toarray()

data_te = CountVectorizer(vocabulary=count_vectorizer.vocabulary_).fit_transform(data_te)
X_te = TfidfTransformer().fit_transform(data_te.toarray()).toarray()

# 贝叶斯模型
model_gau = GaussianNB()
model_gau.fit(X_tr, labels_tr)
# model_gau.score(X_te, labels_te)
label_true_gau, y_pred_gau = labels_te, model_gau.predict(X_te)
print(classification_report(label_true_gau, y_pred_gau))


# 支持向量分类模型，默认参数
model_svc = SVC()
model_svc.fit(X_tr, labels_tr)
# model_svc.score(X_te, labels_te)
labels_true_svc, y_pred_svc = labels_te, model_svc.predict(X_te)
print(classification_report(labels_true_svc, y_pred_svc))

# 基于网格搜索调参
param = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000]}

model_svc_val = GridSearchCV(SVC(), param_grid=param)
model_svc_val.fit(X_tr, labels_tr)
# model_svc_val.score(X_te, labels_te)
print(model_svc_val.best_params_)
labels_true_svc_val, y_pred_svc_val = labels_te, model_svc_val.predict(X_te)
print(classification_report(labels_true_svc_val, y_pred_svc_val))



