import numpy as np
from random import random
from igraph import *
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append("..")     
from utils import *

def Greedy_ICM(G,k,num_iterations):
    '''
    G: igraph graph object used for greedy algorithm
    k: seed set size
    num_iterations: num of iterations 
    '''

    seed_set = []

    while len(seed_set) < k:
        
        cascading_power = {}
        for node in (G.vs):
            if node.index not in seed_set:
                cascading_power[node.index] = 0
                
                for i in range(num_iterations):
                    seed_copy = seed_set.copy()
                    seed_copy.append(node.index)

                    cascaded = ICM(G,seed_copy)
                    cascading_power[node.index] += len(cascaded)
                
                cascading_power[node.index] = cascading_power[node.index]/num_iterations
        
        #get node with max power
        max_node = max(cascading_power,key = cascading_power.get)
        seed_set.append(max_node)
        inf_size = cascading_power[max_node]
    
    return seed_set,inf_size

df_results = pd.DataFrame(None,columns=["Dataset","Original","Degree",'RA','AA','LHI','Jaccard'])
for data in [("USairport500.txt","d","w"),("OClinks_w.txt","d","w"),("USairport_2010.txt","d","w")]:

# ("celegans_n306.txt","d","w"),("USairport500.txt","d","w"),
# ("Freemans_EIES-3_n32.txt","d","w"),

# ("USairport500.txt","d","w"),
# ("Freemans_EIES-3_n32.txt","d","w"),("OClinks_w.txt","d","w"),
# ("USairport_2010.txt","d","w")
   
    dataset,directed,weighted = data

    print(f"--------{dataset}---------")
    
    idx = len(df_results.index)
    
    df_results.at[idx,"Dataset"] = dataset
    
    seed_size = 5
    num_iter = 20

    if dataset in ["email.txt" ,"road.txt" ,"weblog.txt" ,"animal_interaction.txt","facebook_combined.txt"]:
        data_file_path = os.path.join(os.pardir,os.pardir,"data","Prev Data",dataset)
    else:
        data_file_path = os.path.join(os.pardir,os.pardir,"data",dataset)
    
    directed = True if directed == "d" else False
    weighted = True if weighted == "w" else False

    G = read_graph(data_file_path,directed,weighted = weighted)

    G.es['p'] = scale(get_true_weights(dataset,directed,data_file_path,weighted))
    original_results = Greedy_ICM(G,seed_size,num_iter)
    
    df_results.at[idx,"Original"] = original_results
    print("Original ",original_results)

    G.es['p'] = compute_prob(G)
    degree_results = Greedy_ICM(G,seed_size,num_iter)
    df_results.at[idx,"Degree"] = degree_results
                  
    print("Degree ",degree_results)

    p = ['RA','AA','LHI','Jaccard']


    for heuristic in p:
        G.es['p'] = prob_heuristics(G,heuristic)
        heuristic_result = Greedy_ICM(G,seed_size,num_iter)
        
        df_results.at[idx,heuristic] = heuristic_result
        
        print(heuristic,heuristic_result)

    print("-------------------")
    
    df_results.to_csv("Greedy_op.csv",index=False)