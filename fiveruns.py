# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import pickle
import pandas as pd
import numpy as np


experiment_name = 'BestSolution_Gaussian_Enemy6'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 20

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="fastest",
                  enemymode="static",
                  level=2)


#test = np.load('EAgaus/enemy6/exp{}_bestsolution.npy'.format(best_solution))

#ACCESS CORRECT FOLDER
eas = ['uni','gaus']
enemies = [2,5,6]

for enemy in enemies:
    for ea in eas:
        folder = 'EA' + ea + '/enemy%i'%enemy
        #Update the enemy
        env.update_parameter('enemies',[enemy])
        seed = np.linspace(1,10,10,dtype = 'int')
        #running 5 times
        for seed in seed:
            test = np.load(folder + '/exp%i'%seed + '_bestsolution.npy')
            data = {'pe':[],'ee':[],'gain':[]}
            for i in range(5):
                print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(enemy)+' \n')
                _,playerenergy,enemyenergy,_ = env.play(pcont=np.array(test))
                data['pe'].append(playerenergy)
                data['ee'].append(enemyenergy)
                data['gain'].append(playerenergy-enemyenergy)
            pickle.dump(data, open(folder+ '/exp%i'%seed + '_gains.p',"wb"))