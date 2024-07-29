import numpy as np
from random import random
from igraph import *
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append("..")     
from utils import *

def decay_function(p,t):
    '''
    Returns f(p,t) where f is the decay function 
    '''
    return p/np.sqrt(t+1)

def ICM_decay(G,start_nodes,decay_fn):
    uninfected = [v for v in G.vs.indices if v not in start_nodes]
    infected = start_nodes.copy()
    active = start_nodes.copy()

    t = 0
    while len(active)!=0 and len(uninfected)!=0: #each iteration of while is one time step
        reached = []
        for node in active:# for each active node 
            for neighbor in G.neighbors(node,mode= 'out'): #each neighbour
                if neighbor in uninfected:   # try to infect each uninfected node 
                    if random() <= decay_fn(G.es[G.get_eid(node,neighbor)]['p'],t):   # gen random number ; if num <= p' p = f(p,t) decay
                        # cost += G.vs[neighbor]['cost']
                        infected.append(neighbor)
                        reached.append(neighbor)
                        uninfected.remove(neighbor)
                        #==> infected , update reached 
                    # else failed

        active = reached.copy()#set active for next iteration as reached
        t +=1
    
    return infected


for data in  [("email.txt","u") ,("road.txt","u") ,("weblog.txt","u")
      ,("animal_interaction.txt","u"),("zachary.graphml","u")]:
    #("facebook_combined.txt","u")("zachary.graphml","u")

    dataset,directed = data
    if dataset in ["email.txt" ,"road.txt" ,"weblog.txt" ,"animal_interaction.txt","facebook_combined.txt"]:
        data_file_path = os.path.join(os.pardir,os.pardir,"data","Prev Data",dataset)
    else:
        data_file_path = os.path.join(os.pardir,os.pardir,"data",dataset)
    
    directed = True if directed == "d" else False
    G = read_graph(data_file_path,directed)

    # print(prob_heuristics(G,'Jaccard'))

    num_iter = 20
    p = ['RA','AA','LHI','Jaccard'] #

    df_cascade = pd.DataFrame(None)
    df_cascade['Node'] = G.vs.indices
    df_cascade['out_core'] = G.coreness(mode='out')

    for prob in p:
        G.es['p'] = prob_heuristics(G,prob)
        # print(G.es['p'])
        
        cascade = []
        
        for seed in tqdm(G.vs.indices): #for each node
            power_n = []
            for i in (range(num_iter)): #go over all worlds
                nodes_i = ICM_decay(G,start_nodes=[seed],decay_fn = decay_function) 
                power_n.append(nodes_i) # get all cascades in each trial
            cascade.append(power_n) # add all cascades of a node to global cascade
        
        df_cascade[f'p={prob}_decay'] = cascade

    df_cascade.to_csv(os.path.join("Cascade outputs",f"{dataset.split('.')[0]}_heuristics_decay_root_t.csv"),index=False)