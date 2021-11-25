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
    X1 = np.reshape(x_v, (4,4))
    fig=sns.heatmap(X1,  cmap="YlGnBu", annot=True, cbar=False, square=True)
    plt.tight_layout()
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) 
    fig.figure.savefig('results/'+task+'.png')

#1. LEFT: 0
#     2. DOWN = 1
#     3. RIGHT = 2
#     4. UP = 3
    
P_fl=hiive.mdptoolbox.openai.OpenAI_MDPToolbox("FrozenLake-v1").P
R_fl=hiive.mdptoolbox.openai.OpenAI_MDPToolbox("FrozenLake-v1").R


###Value iteration 
task='FL-value'
fl_value = hiive.mdptoolbox.mdp.ValueIteration(P_fl,R_fl,0.999,10**(-10),10**5,skip_check=True)
fl_value.run()

fl_value_results =pd.DataFrame.from_dict(fl_value.run_stats)

print("max mean v", max(fl_value_results["Max V"]))

print("mean mean v", max(fl_value_results["Mean V"]))
fl_value_results.to_csv(task+'.csv',index=False)

plot_grid(fl_value.V,'FL_value_value')
plot_grid(fl_value.policy,'FL_value_policy')

####
task='FL-policy'
fl_policy = hiive.mdptoolbox.mdp.PolicyIteration(P_fl,R_fl,0.999, max_iter = 10**5, skip_check=True)
fl_policy.run()
fl_policy_results =pd.DataFrame.from_dict(fl_policy.run_stats)

print("max mean v", max(fl_policy_results["Mean V"]))
fl_policy_results.to_csv(task+'.csv',index=False)

plot_grid(fl_policy.V,'FL_policy_value')
plot_grid(fl_policy.policy,'FL_policy_policy')


plot_results(fl_value_results["Iteration"], fl_value_results["Mean V"],fl_policy_results["Iteration"], fl_policy_results["Mean V"], 
                "Mean Value", 'FL-value')

plot_results(fl_value_results["Iteration"], fl_value_results["Max V"],fl_policy_results["Iteration"], fl_policy_results["Max V"], 
                "Reward", 'FL-value-max')



plot_results(fl_value_results["Iteration"], fl_value_results["Time"], fl_policy_results["Iteration"], fl_policy_results["Time"], 
                  "Time (s)",'FL-time' )


for esp in [1,0.9,0.8,0.5]:
    task='FL-q'+str(esp)
    #fl_q = hiive.mdptoolbox.mdp.QLearning(P_fl, R_fl, 0.999, epsilon=esp, n_iter=10**5, alpha=0.95, skip_check=True)
    fl_q = hiive.mdptoolbox.mdp.QLearning(P_fl, R_fl, 0.999, epsilon=1,epsilon_decay=0.999, n_iter=10**6, alpha=0.6,alpha_decay=1, skip_check=True)
    fl_q.run()

    fl_q_results = pd.DataFrame.from_dict(fl_q.run_stats)

    print("max mean v", max(fl_q_results["Mean V"]))
    fl_q_results.to_csv(task+'.csv',index=False)


    plot_Q(fl_q_results["Iteration"], fl_q_results["Mean V"],str(esp),"Mean Value", 'Q-max-value'+str(esp))
    plot_Q(fl_q_results["Iteration"], fl_q_results["Max V"],str(esp),"Max Value", 'Q-value'+str(esp))
    plot_Q(fl_q_results["Iteration"], fl_q_results["Time"],str(esp),"Time (s)", 'Q-time'+str(esp))


plot_grid(fl_q.V,'FL_q_value')
plot_grid(fl_q.policy,'FL_q_policy')

task='Reward'
for esp in [1,0.9,0.8,0.5]:
    fl_q_results=pd.read_csv('FL-q'+str(esp)+'.csv')
    
    
    plt.xlabel('Iteration')
    plt.ylabel(task)
    x_var_v, y_var_v=fl_q_results["Iteration"], fl_q_results["Max V"]
    plt.plot(x_var_v, y_var_v, 'o-',label=str(esp))
 #   plt.plot(x_var_p, y_var_p, 'o-',label='Policy Iteration')
    
plt.legend(fontsize=12)
plt.xscale('log')
plt.tight_layout()

plt.savefig('results/'+task+'.pdf',format='pdf',dpi=400)
    

