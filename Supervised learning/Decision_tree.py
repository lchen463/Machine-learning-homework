import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score 
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_breast_cancer

from util import plot_confusion_matrix,plot_learning_curve
task='Decision_tree'


cancer = load_breast_cancer()
data_brest = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names, ["target"])
data= pd.DataFrame(data_brest, columns=columns)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

###Normalization

print()
####Tuning max_depts using 5-CV


def plot_feature_importances(clf, feature_names):
    c_features = len(feature_names)
    plt.barh(range(c_features), clf.feature_importances_)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.yticks(np.arange(c_features), feature_names)
    plt.savefig('feature.pdf', fmt='pdf',dpi=300)

def plot_cv(variable,score,task):
    plt.figure()
    plt.plot(variable,score,'o-')
    plt.xlabel("Max depth")
    plt.ylabel("Average cross validation score")
    plt.tight_layout()
    
    plt.savefig(task+'CV.pdf', fmt='pdf',dpi=300)




feature_reduction=0

from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
scaler = StandardScaler()

import time
start_time = time.time()



print (X.shape)

max_depth = [1,2,3,4,5,6]

leaf=[8,10,12,14,16]

score_ini=0.0
score_cv=[]
for val in max_depth:
  #  for l in leaf:
        X=scaler.fit_transform(X)
        score = cross_val_score(DecisionTreeClassifier(max_depth= val, random_state= 0), X, y, cv= 5, scoring="accuracy")
        score_mean=score.mean()
        score_cv.append(score_mean)
            
        print(f'Average score({val}): {"{:.3f}".format(score.mean())}')
            
plot_cv(max_depth,score_cv,task)  
  
          
        
 ####Best=max_depth=4, min_samples_leaf=8   
    

train_size=[0.1,0.2,0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]    


cv_scores=[]

seeds=[0,1000,30000,2000000,4000000]


train_scores_mean=[]

train_scores_std=[]

test_scores_mean=[]

test_scores_std=[]

for size in train_size:    
   train_scores=[]
   test_scores=[]
    
   for seed in seeds:

    
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=1-size,random_state=seed)
        
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.fit_transform(X_test)        
        
        clf1 =DecisionTreeClassifier(max_depth= 2, random_state= seed)
        
        
        clf1.fit(X_train, Y_train)
        
     
        
        Train_score=clf1.score(X_train, Y_train)
        Test_score=clf1.score(X_test, Y_test)
         
        train_scores.append(Train_score)
        test_scores.append(Test_score)
    
   train_scores_mean.append(np.mean(train_scores))
   train_scores_std.append(np.std(train_scores))
   test_scores_mean.append(np.mean(test_scores))
   test_scores_std.append(np.std(test_scores))    


train_sizes=np.array(train_size)*len(X)


    
    
results=pd.DataFrame()


results['train_sizes']=train_sizes
results['train_scores_mean']=train_scores_mean
results['train_scores_std']=train_scores_std
results['test_scores_mean']=test_scores_mean
results['test_scores_std']=test_scores_std

results.to_csv(str(task)+'.csv',index=False)

#np.savetxt(str(task)+'.txt',[train_sizes,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std])   
plot_learning_curve(train_sizes,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std,'Learning Curve',task)


##confusion matrix:
    
    
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1,random_state=0)

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

clf1 =DecisionTreeClassifier(max_depth= 2, random_state= 0)


clf1.fit(X_train, Y_train)

prediction = clf1.predict(X_test)
cnf_matrix = confusion_matrix(Y_test, prediction)

dict_characters = {0: 'Malignant', 1: 'Benign'}
plot_confusion_matrix(cnf_matrix, classes=dict_characters,feature=task,title='Confusion matrix')

print("--- %s seconds ---" % (time.time() - start_time))