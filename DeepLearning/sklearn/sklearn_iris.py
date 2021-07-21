from sklearn.linear_model import LinearRegression, LogisticRegression

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import roc_curve

import numpy as np
import pandas as pd
iris = load_iris()

# 실습

iris_label = iris.target
print(type(iris))
print(iris.keys())
print(iris['data'].shape, iris['target'].shape)
X_train, X_test, y_train, y_test = train_test_split(
                                                    iris['data'],
                                                    iris['target'],
                                                    test_size=0.2,
                                                    stratify=iris['target'],
                                                    random_state=0
                                                    )
print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)

print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)


logreg = LogisticRegression().fit(X_train,y_train)


print('Training set score : ',logreg.score(X_train,y_train))
print('Test set score : ',logreg.score(X_test,y_test))

y_pred = logreg.fit(X_train,y_train).predict(X_test)

result = []
print(y_pred,'\n',y_test)
for p,t in zip(y_pred,y_test):
    if p == t:
        result.append(1)
    else:
        result.append(0)

print('result', result)

probs = logreg.predict_proba(X_test)
print(probs.shape,probs[:,1])

#
# fpr, tpr, thresholds = roc_curve(y_test, probs[:,1], pos_label=2)
# plt.plot(fpr, tpr, label="ROC Curve")
# plt.xlabel("FPR")
# plt.ylabel("TPR (recall)")
# # find threshold closest to zero
# close_zero = np.argmin(np.abs(thresholds))
# plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
# label="threshold zero", fillstyle="none", c='k', mew=2)
# plt.legend(loc=4)
#
# plt.show()