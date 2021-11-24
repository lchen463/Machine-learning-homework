###Ref: https://www.kaggle.com/josephstalinpeter/clustering-algorithms-breast-cancer-wiscosin


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
import sys




from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_breast_cancer

from sklearn.metrics import accuracy_score
from util import plot_confusion_matrix,plot_learning_curve

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA,FastICA
from sklearn.metrics import silhouette_score,v_measure_score

from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

from sklearn.random_projection import GaussianRandomProjection

def plot_cv(variable,score,sse_,task):
    plt.figure(0)
    plt.plot(variable,score,'o-',label='v_measure score')
    plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1],'o--',label='Silhouette_score')
    plt.xlabel("Number of clusters (K)")
    
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(task+'K-score.pdf', fmt='pdf',dpi=300)

def  plot_comp(X, y_predict,y_true,xlabel,ylabel,task):
    plt.figure(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    ax1.scatter(X[:,0],X[:,1],  c=y_predict, cmap = "jet", edgecolor = "None", alpha=0.35)
    ax1.set_title('GM clustering plot')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    
    
    ax2.scatter(X[:,0],X[:,1],  c = y_true, cmap = "jet", edgecolor = "None", alpha=0.35)
    ax2.set_title('Actual clusters')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)    
    plt.savefig(task+'best-compare.pdf', fmt='pdf',dpi=300)

def plot_pca(variable,score,sse_,task):
    plt.figure(2)
    plt.plot(variable,score,'o-',label='v_measure score')
    plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1],'o--',label='Silhouette_score')
    plt.xlabel("Number of components")
    
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(task+'K-score.pdf', fmt='pdf',dpi=300)
    

import sys
task='GM_GRP'

cancer = load_breast_cancer()
data_brest = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names, ["target"])
data= pd.DataFrame(data_brest, columns=columns)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

scaler = StandardScaler()

X=scaler.fit_transform(X)

#print("--- %s seconds ---" % (time.time() - start_time))

# 2. Silhouette Analysis

if task =='GM':
    best_score=0
    
    for j in ['spherical','diag','tied','full']:
        scores=[]
        sse_ = []    
        for K in range(2,18,2):
        
            
            GM=GaussianMixture(n_components=K, covariance_type=j,random_state=0)
            
            kY = GM.fit_predict(X)
            
        
            scores.append(v_measure_score(kY,y))
            
            sse_.append([K, silhouette_score(X, kY)])
            
            
            if np.max(scores)==v_measure_score(kY,y):
                y_predict=kY
                best_score=np.max(scores)
            
        plt.figure(0)
        plt.plot(np.arange(2,18,2),scores,'o-',label=j+' v_measure score')
        plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1],'s-',label=j+' Silhouette_score')
        plt.xlabel("Number of clusters (K)")
        
        plt.ylabel("Score")
        
        plt.ylim(0,1)
        plt.legend(ncol=1,loc=1,bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.savefig(task+'-score.pdf', fmt='pdf',dpi=300)
    #plot_sa(sse_,task)
    
    plot_comp(X, y_predict,y,'mean radius','mean texture',task)
    
    print (best_score)
###


#####



if task=='GM_PCA':
    
    scores=[]
    sse_ = [] 
    for n in range(2,10):
        
        pca=PCA(n_components=n)
        pca.fit(X)
        
        X_pca=pca.transform(X)
        
        
        
        GM=GaussianMixture(n_components=2, covariance_type='full',random_state=0)
        
        Y_pca = GM.fit_predict(X_pca)
        
        print (pca.explained_variance_ratio_)
        print (pca.explained_variance_)    
    
        scores.append(v_measure_score(y,Y_pca))
        sse_.append([n, silhouette_score(X, Y_pca)])
        
        if np.max(scores)==v_measure_score(Y_pca,y):
            y_predict=Y_pca
            best_score=np.max(scores)
            
    
    plot_pca(np.arange(2,10),scores,sse_,task)
    
    print (best_score)
    
    
    plot_comp(X_pca, Y_pca,y,'X0','X1',task)



if task=='GM_ICA':
    
    scores=[]
    sse_ = [] 
    
    for n in range(2,10):
        
        ica=FastICA(n_components=n)
        ica.fit(X)
        
        X_ica=ica.transform(X)
        
        
        GM=GaussianMixture(n_components=2, covariance_type='full',random_state=0)
        
        Y_ica = GM.fit_predict(X_ica)
    
    
        scores.append(v_measure_score(Y_ica,y))
        sse_.append([n, silhouette_score(X, Y_ica)])
        
        if np.max(scores)==v_measure_score(Y_ica,y):
            y_predict=Y_ica
            best_score=np.max(scores)
            
    
    plot_pca(np.arange(2,10),scores,sse_,task)
    
    print (best_score)
    
    
    plot_comp(X_ica, Y_ica,y,'X0','X1',task)


if  task=='GM_GRP':
    for seed in [0,100,200]:
        scores=[]
        sse_=[]
        for n in range(2,10):
                
            rp= GaussianRandomProjection(n_components=n,random_state=seed)
            X_rp=rp.fit_transform(X)
            
            GM=GaussianMixture(n_components=2, covariance_type='full',random_state=0)
            
            
            Y_rp = GM.fit_predict(X_rp)
        
        
           
            
            hs=v_measure_score(Y_rp,y)
            sse_.append([n, silhouette_score(X, Y_rp)])
            
            scores.append(hs)
            if np.max(scores)==hs:
                y_predict=Y_rp
                best_score=np.max(scores)
                
        
        plot_pca(np.arange(2,10),scores,sse_,task)
    
    print (best_score)
    
    
    plot_comp(X_rp, Y_rp,y,'X0','X1',task)

if  task=='GM_TSNE':
    scores=[]
    sse_=[]
    
    for n in [2,3]:
            
        tsne=TSNE(n_components=n,random_state=0)
        X_tsne=tsne.fit_transform(X)
        
        GM=GaussianMixture(n_components=2, covariance_type='full',random_state=0)
        
        Y_tsne =GM.fit_predict(X_tsne)
    
        hs=v_measure_score(Y_tsne,y)
        
        scores.append(hs)
        sse_.append([n, silhouette_score(X, Y_tsne)])
        if np.max(scores)==hs:
            y_predict=Y_tsne
            best_score=np.max(scores)
            
    
    plot_pca([2,3],scores,sse_,task)
    
    print (best_score)
    
    
    plot_comp(X_tsne, Y_tsne,y,'X0','X1',task)





