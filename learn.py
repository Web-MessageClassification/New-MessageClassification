# sklearn 文本分类学习代码：
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# first extract the 20 news_group dataset to /scikit_learn_data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

# all categories
# newsgroup_train = fetch_20newsgroups(subset='train')
# part categories
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med'];
twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories, shuffle=True, random_state=42)
twenty_train.target_names
['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
print(len(twenty_train.data))
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])
print(twenty_train.target[:20])
for t in twenty_train.target[:20]:
    print(twenty_train.target_names[t])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'algorithm'))

tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicted = clf.predict(X_new_tfidf)

print('test result')
# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, twenty_train.target_names[category]))


text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data,twenty_train.target)
# new_predicted = text_clf.predict(X_new_tfidf)

#Evaluation of the performance on the test set
print('MultinomialNB')
twenty_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

#SVM
print('SVM')
text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',SGDClassifier(loss='hinge',alpha=1e-3,n_iter=5,random_state=42)),])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

print (metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))


