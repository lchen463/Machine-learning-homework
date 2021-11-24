#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:30:42 2021

@author: lihua

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import itertools


def plot_learning_curve(train_sizes,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std,title,feature):
    
    train_sizes=train_sizes.astype(int)
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
    
    
    

    plt.xticks(train_sizes)
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
    
