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

def get_true_weights(dataset : str,directed : str,data_file_path : str,weighted = True) -> list:
    '''
    
    Inputs:
    dataset : name of dataset
    directed : True/False 
    data_file_path : path containing original dataset so weights can be generated if weights csv is not found in current path
    weighted = True/False
    '''
    extension = data_file_path.split('.')[-1]

    
    if extension == "graphml":
        G = Graph.Read_GraphML(data_file_path,directed)
    elif extension == "txt" or extension == 'edgelist':
        try:
            # print("reading weights")
            G = Graph.Read_Ncol(data_file_path,names = False,directed = directed,weights = weighted)
            
        except:
            print("read edgelist")
            G = Graph.Read_Edgelist(data_file_path,directed)
    elif extension == "csv":
        dataframe  = pd.read_csv(data_file_path)
        G = Graph.DataFrame(dataframe,directed = directed)
    else:
        print("Input file format not supported")
        return
    
    if directed == False:
        print("Undirected graph; Input directed graph")
        return
    

    G.simplify(loops=True,combine_edges = "sum") #remove self loops and combine multiple edges into single edge with attributes summed up
   
    G.delete_vertices(G.vs.select(_degree = 0 )) #remove vertices with zero edges
   
    return G.es['weight']

for data in  [("OClinks_w.txt","d","w")]:


    #("celegans_n306.txt","d","w"),("USairport500.txt","d","w")
            # ,("openflights.txt","d","w"),("USairport_2010.txt","d","w"),
    # ("Freemans_EIES-3_n32.txt","d","w")("OClinks_w.txt","d","w")

    # ,("lkml.txt","d","w"),("soc-redditHyperlinks-body.txt","d","w"),("soc-redditHyperlinks-title.txt","d","w")


    dataset,directed,weighted= data
    if dataset in ["email.txt" ,"road.txt" ,"weblog.txt" ,"animal_interaction.txt","facebook_combined.txt"]:
        data_file_path = os.path.join(os.pardir,os.pardir,"data","Prev Data",dataset)
    else:
        data_file_path = os.path.join(os.pardir,os.pardir,"data",dataset)
    
    directed = True if directed == "d" else False
    weighted = True if weighted == "w" else False

    G = read_graph(data_file_path,directed,weighted = weighted)

    # print(prob_heuristics(G,'Jaccard'))

    num_iter = 20
    

    df_cascade = pd.DataFrame(None)
    df_cascade['Node'] = G.vs.indices
    df_cascade['out_core'] = G.coreness(mode='out')
    # df_prob = pd.DataFrame(None)

    G.es['p'] = scale(get_true_weights(dataset,directed,data_file_path,weighted))
    # df_prob[prob] = G.es['p']
    # print(G.es['p'])
    
    cascade = []
    
    for seed in tqdm(G.vs.indices): #for each node
        power_n = []
        for i in (range(num_iter)): #go over all worlds
            nodes_i = ICM(G,start_nodes=[seed]) 
            power_n.append(nodes_i) # get all cascades in each trial
        cascade.append(power_n) # add all cascades of a node to global cascade
    
    df_cascade['p=Original'] = cascade

    df_cascade.to_csv(os.path.join("Cascade outputs",f"{dataset.split('.')[0]}_original.csv"),index=False)
    # df_prob.to_csv(f"{dataset.split('.')[0]}_weights.csv",index=False)


    #---------#
    # decay cascade 
    #---------#
    
    df_cascade = pd.DataFrame(None)
    df_cascade['Node'] = G.vs.indices
    df_cascade['out_core'] = G.coreness(mode='out')
    
    cascade = []
    
    for seed in tqdm(G.vs.indices): #for each node
        power_n = []
        for i in (range(num_iter)): #go over all worlds
            nodes_i = ICM_decay(G,start_nodes=[seed],decay_fn = decay_function) 
            power_n.append(nodes_i) # get all cascades in each trial
        cascade.append(power_n) # add all cascades of a node to global cascade
    
    df_cascade['p=Original_decay'] = cascade

    df_cascade.to_csv(os.path.join("Cascade outputs",f"{dataset.split('.')[0]}_original_decay_root_t.csv"),index=False)