#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

"""
print(__doc__)



# Code Main source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

# Modified: Saboora M. Roshan

import numpy as np
import codecs
import arff


"""   
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
    """ 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=25),
    SVC(gamma="auto", C=10),
    #GaussianProcessClassifier(ConstantKernel(0.1, (0.01, 10.0))
     #          * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)) ** 2), max_iter_predict=100,warm_start=True),
    GaussianProcessClassifier(1.0 * RBF(1.0), max_iter_predict=100,warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=50, n_estimators=1000, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(n_estimators=500),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
"""   
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
   """
datasets = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
"""   
            make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
               """
            
#file_ = codecs.load('messidor_features.arff', 'rb', 'utf-8')
data=arff.loads(open('messidor_features.arff','rb'))     
new=data['data']
newarray=np.asarray(new)
start=newarray[:,:19]
end=newarray[:,19:]
endy=end.astype(float)
starty=start.astype(float)
i=0
for xval in endy:
    if xval<1:
        endy[i]=-1
    i+=1

"""    
figure = plt.figure(figsize=(27, 9))
    """ 
i = 1
# iterate over datasets
#for ds_cnt, ds in enumerate(datasets):
# preprocess dataset, split into training and test part
X, y = datasets
#print X.size
#print y.size

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
"""    
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    """ 
startynorm=StandardScaler().fit_transform(starty)
Xn_train, Xn_test, yn_train, yn_test = \
    train_test_split(startynorm, endy, test_size=.3, random_state=42)
# just plot the dataset first
"""   
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
if ds_cnt == 0:
    ax.set_title("Input data")
    
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())   
   """
i += 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    """   
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        """ 
#    clf.fit(X_train, y_train)
#    score = clf.score(X_test, y_test)
#    print name
#    print score
#    print precision_recall_fscore_support(y_test,clf.predict(X_test),average="macro")
    clf.fit(Xn_train, np.ravel(yn_train))
    score = clf.score(Xn_test, yn_test)
    print name
    print score
    print precision_recall_fscore_support(yn_test,clf.predict(Xn_test),average="macro")
   
    
    """    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_cnt == 0:
        ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    
        """ 
    i += 1
"""      
plt.tight_layout()
plt.show()
    """ 