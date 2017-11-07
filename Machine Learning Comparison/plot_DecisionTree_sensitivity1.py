
print(__doc__)


import numpy as np
import codecs
import arff
import pandas as pd

 
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

names=[]
classifiers=[]

#count=list(range(1,101,0.5))
count=np.arange(1,100,1)
for k in count:
    names.append("Decision Tree " + str(k))
    classifiers.append(DecisionTreeClassifier(max_depth=k))

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


startynorm=StandardScaler().fit_transform(starty)
Xn_train, Xn_test, yn_train, yn_test = \
    train_test_split(startynorm, endy, test_size=.3, random_state=42)
# just plot the dataset first

#i += 1
result=[]
# iterate over classifiers
for name, clf in zip(names, classifiers):


#    print precision_recall_fscore_support(y_test,clf.predict(X_test),average="macro")
    clf.fit(Xn_train, np.ravel(yn_train))
    score = clf.score(Xn_test, yn_test)
#    print name
#    print score
    res=precision_recall_fscore_support(yn_test,clf.predict(Xn_test),average="macro")
#    print res
    result.append([name,score,res])
    
 
    #i += 1

#np.savetxt("KnearestNeighbor.txt", result)
my_df = pd.DataFrame(result)
my_df.to_csv('Decision_tree.csv', index=False, header=False)