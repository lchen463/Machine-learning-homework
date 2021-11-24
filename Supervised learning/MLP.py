import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier


from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score 
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_breast_cancer

from util import plot_confusion_matrix,plot_learning_curve
task='MLP'

cancer = load_breast_cancer()
data_brest = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names, ["target"])
data= pd.DataFrame(data_brest, columns=columns)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]




import tensorflow as tf

###Normalization

import time
start_time = time.time()



scaler = StandardScaler()




print()
####Tuning max_depts using 5-CV


def plot_feature_importances(clf, feature_names):
    c_features = len(feature_names)
    plt.barh(range(c_features), clf.feature_importances_)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.yticks(np.arange(c_features), feature_names)
    plt.savefig('feature.pdf', fmt='pdf',dpi=300)

def plot_cv(variable,score):
    plt.figure()
    plt.plot(variable,score,'o-')
    plt.xlabel("Number of neighbors")
    plt.ylabel("Average cross validation score")
    
    plt.savefig('CV.pdf', fmt='pdf',dpi=300)

def plot_learning_curve(train_sizes,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std,title,feature):
    
    
    plt.figure(0)
    plt.title(title)
 #   if ylim is not None:
  #      plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, np.array(train_scores_mean) - np.array(train_scores_std),
                      np.array(train_scores_mean) + np.array(train_scores_std), alpha=0.1,
                      color="r")
    plt.fill_between(train_sizes, np.array(test_scores_mean) - np.array(test_scores_std),
                      np.array(test_scores_mean) + np.array(test_scores_std), alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
   
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(feature)+title+'.pdf',format='pdf',dpi=300)
    return plt

def plot_confusion_matrix(cm, classes,feature,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(str(feature)+title+'.pdf',format='pdf',dpi=300)
    

task='MLP'##best_neighbor 5



#if task==1:
  
 #   clf=DecisionTreeClassifier(max_depth=4,min_samples_leaf=8)
  #  clf.fit(X,y)
    
  #  key_feature=np.nonzero(clf.feature_importances_)
    
  #  name=X.columns.values[key_feature]
    
    
 #   plot_feature_importances(clf, X.columns.values)
  #  X=X[name]


#X=X[key_feature_name]


print (X.shape)

# vals = ['linear','poly','rbf']

# score_ini=0.0

# score_cv=[]


# for val in vals:
        

#         X = scaler.fit_transform(X)
        
        
#         score = cross_val_score(SVC(kernel=val), X, y, cv= 5, scoring="accuracy")
#         score_mean=score.mean()
#         score_cv.append(score_mean)
            
#         print(f'Average score({val}): {"{:.3f}".format(score.mean())}')
        
#  ####Best=max_depth=4, min_samples_leaf=8   

# plot_cv(vals,score_cv)  
  

##confusion matrix:
    

test_scores=[]
    
train_scores=[]
       
#for lc in [0.1]:   ###learning-rate, best 0.01

lc=2
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1,random_state=1000)



X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)



clf1 =tf.keras.Sequential([    
  tf.keras.layers.Dense(20, activation=tf.nn.sigmoid, input_shape=(X_train.shape[1],)), # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
  tf.keras.layers.Dense(2)
])



optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)



clf1.compile(optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

Y_train=np.array(Y_train)
history =clf1.fit(X_train, Y_train,validation_split=0.2,  epochs=10)

Y_test=np.array(Y_test)

test_score=clf1.evaluate(X_test,Y_test)

test_scores.append(test_score[1])
train_scores.append(history.history['accuracy'][-1])


plt.figure(0)
print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'],label='hidden layer '+str(lc)+', train')
plt.plot(history.history['val_loss'],label='hidden layer: '+str(lc)+', validation')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc=1)
plt.savefig(task+'loss.pdf',dpi=300)

plt.figure(2)

# summarize history for loss


plt.plot(history.history['accuracy'],label='hidden layer: '+str(lc)+', train')
plt.plot(history.history['val_accuracy'],label='hidden layer: '+str(lc)+', validation')

plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc=4)
plt.savefig(task+'_accuracy.pdf',dpi=300)

Y_test=np.array(Y_test)

prediction = clf1.predict_classes(X_test)
cnf_matrix = confusion_matrix(Y_test, prediction)

dict_characters = {0: 'Malignant', 1: 'Benign'}
plot_confusion_matrix(cnf_matrix, classes=dict_characters,feature=task,title='Confusion matrix')


print("--- %s seconds ---" % (time.time() - start_time))

print (test_scores,history.history['val_accuracy'][-1])