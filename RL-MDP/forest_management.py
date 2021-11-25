#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:55:56 2021

@author: lihua
"""

#from hiive import mdptoolbox

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import hiive.mdptoolbox 
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example
#import mdptoolbox, mdptoolbox.example
import gym
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import time

os.makedirs('results',exist_ok=true)

def plot_results(x_var_v, y_var_v, x_var_p, y_var_p, value,task):
    plt.figure(figsize=(6,5))
#    plt.rcParams(fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel(value)
    
    plt.plot(x_var_v, y_var_v, 'o-',label='Value Iteration')
    plt.plot(x_var_p, y_var_p, 'o-',label='Policy Iteration')
    
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.tight_layout()

    plt.savefig('results/'+task+'.pdf',format='pdf',dpi=400)
    
def plot_Q(x_var_v, y_var_v,espilon,value,task):
    plt.figure(figsize=(6,5))
#    plt.rcParams(fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel(value)
    
    plt.plot(x_var_v, y_var_v, 'o-',label=str(espilon))
 #   plt.plot(x_var_p, y_var_p, 'o-',label='Policy Iteration')
    
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.tight_layout()

    plt.savefig('results/'+task+'.pdf',format='pdf',dpi=400)
def plot_grid(x_v,task):
    plt.figure(figsize=(8, 8))
    X1 = np.reshape(x_v, (1,2))
    fig=sns.heatmap(X1,  cmap="YlGnBu", annot=True, cbar=False, square=True)
    plt.tight_layout()
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) 
    fig.figure.savefig('results/'+task+'.png')
    
def plot_map_V(x_var_v, y_var_v, x_var_p, y_var_p, value,task):
    plt.figure(figsize=(6,5))
#    plt.rcParams(fontsize=14)
    plt.xlabel('States')
    plt.ylabel(value)
    
    plt.plot(x_var_v, y_var_v, 'o-',label='Value Iteration')
    plt.plot(x_var_p, y_var_p, 'o-',label='Policy Iteration')
    #plt.plot(x_var_q, y_var_q, 'o-',label='Q-learning')
    
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.tight_layout()

    plt.savefig('results/'+task+'.pdf',format='pdf',dpi=400)
    

def plot_Q_table(x_var_v, y_var_v, x_var_p, y_var_p, value,task):
    plt.figure(figsize=(6,5))
#    plt.rcParams(fontsize=14)
    plt.xlabel('States')
    plt.ylabel(value)
    
    plt.plot(x_var_v, y_var_v, 'o--',label='Wait')
    plt.plot(x_var_p, y_var_p, 'ro--',alpha=0.7,label='Cut')
    
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.tight_layout()

    plt.savefig('results/'+task+'.pdf',format='pdf',dpi=400)


P, R = hiive.mdptoolbox.example.forest(S=1000, p=0.01)

initial_time=time.time()

task='Forest-value'

forest_value =  hiive.mdptoolbox.mdp.ValueIteration(P, R, 0.999, 10**(-50),10**5,skip_check=True)
forest_value.run()

value_ct=time.time()-initial_time

forest_value_results =pd.DataFrame.from_dict(forest_value.run_stats)

print("mean v", max(forest_value_results["Mean V"]))
print("max  v", max(forest_value_results["Max V"]))
forest_value_results.to_csv('results/'+task+'.csv',index=False)



forest_policy = hiive.mdptoolbox.mdp.PolicyIteration(P,R, 0.999,max_iter =10**5, skip_check=True)
forest_policy.run()

forest_policy_results = pd.DataFrame.from_dict(forest_policy.run_stats)

print("max mean v", max(forest_policy_results["Mean V"]))
forest_policy_results.to_csv('results/'+task+'.csv',index=False)



plot_results(forest_value_results["Iteration"], forest_value_results["Mean V"],forest_policy_results["Iteration"], forest_policy_results["Mean V"], 
                "Mean Value", 'forest-value')

plot_results(forest_value_results["Iteration"], forest_value_results["Max V"],forest_policy_results["Iteration"], forest_policy_results["Max V"], 
                "Reward", 'forest-value-max')



plot_results(forest_value_results["Iteration"], forest_value_results["Time"], forest_policy_results["Iteration"], forest_policy_results["Time"], 
                  "Time (s)",'forest-time' )

plot_map_V(np.arange(0,1000), forest_value.V, np.arange(0,1000), forest_policy.V, 
                  "Value",'forest-value-map' )


plot_map_V(np.arange(0,1000), forest_value.policy, np.arange(0,1000), forest_policy.policy, 
                  "Policy",'forest-policy-map' )


for esp in [1,0.9,0.8,0.5]:
    task='forest-q'+str(esp)
    
    forest_q =  hiive.mdptoolbox.mdp.QLearning(P, R, 0.999, epsilon=esp,epsilon_decay=0.9999, n_iter=10**6, alpha=0.6,alpha_decay=1, skip_check=True)
    forest_q.run()
    forest_q_results = pd.DataFrame.from_dict(forest_q.run_stats)

    print("max mean v", max(forest_q_results["Mean V"]))
    forest_q_results.to_csv('results/'+task+'.csv',index=False)
    
    plot_Q(forest_q_results["Iteration"], forest_q_results["Mean V"],str(esp),"Mean Value", 'Q-value'+str(esp))
    plot_Q(forest_q_results["Iteration"], forest_q_results["Max V"],str(esp),"Max Value", 'Q-value'+str(esp))


task='Time'
for esp in [1,0.9,0.8,0.5]:
    fl_q_results=pd.read_csv('results/forest-q'+str(esp)+'.csv')
    
    
    plt.xlabel('Iteration')
    plt.ylabel('Time (s)')
    x_var_v, y_var_v=fl_q_results["Iteration"], fl_q_results["Time"]
    plt.plot(x_var_v, y_var_v, 'o-',label=str(esp))
 #   plt.plot(x_var_p, y_var_p, 'o-',label='Policy Iteration')
    
plt.legend(fontsize=12)
plt.xscale('log')
plt.tight_layout()

plt.savefig('results/forest'+task+'.pdf',format='pdf',dpi=400)
    

plot_Q(np.arange(0,1000),forest_q.V,'Q-learning',"Value",'Q-value-map')


plot_Q(np.arange(0,1000),forest_q.policy,'Q-learning',
                  "Policy",'Q-policy-map' )

Q_table=forest_q.Q

plot_Q_table(np.arange(0,1000), Q_table[:,0], np.arange(0,1000), Q_table[:,1], 
                  "Q_Table, Value",'Q-table-map')


