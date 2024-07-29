import numpy as np
from random import random
from igraph import *
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append("..")     
from utils import *


for data in  [("OClinks_w.txt","d","w")]:

    #("celegans_n306.txt","d","w"),("USairport500.txt","d","w")
            # ,("openflights.txt","d","w"),("USairport_2010.txt","d","w"),
    # ("Freemans_EIES-3_n32.txt","d","w"),("OClinks_w.txt","d","w")


    # ,("lkml.txt","d","w"),("soc-redditHyperlinks-body.txt","d","w"),("soc-redditHyperlinks-title.txt","d","w")

    dataset,directed,weighted = data
    if dataset in ["email.txt" ,"road.txt" ,"weblog.txt" ,"animal_interaction.txt","facebook_combined.txt"]:
        data_file_path = os.path.join(os.pardir,os.pardir,"data","Prev Data",dataset)
    else:
        data_file_path = os.path.join(os.pardir,os.pardir,"data",dataset)
    
    directed = True if directed == "d" else False
    weighted = True if weighted == "w" else False

    G = read_graph(data_file_path,directed,weighted = weighted)


    # print(prob_heuristics(G,'Jaccard'))

    num_iter = 20
    p = ['RA','AA','LHI','Jaccard'] #

    df_cascade = pd.DataFrame(None)
    df_cascade['Node'] = G.vs.indices
    df_cascade['out_core'] = G.coreness(mode='out')
    df_prob = pd.DataFrame(None)

    for prob in p:
        G.es['p'] = prob_heuristics(G,prob)
        df_prob[prob] = G.es['p']
        # print(G.es['p'])
        
        cascade = []
        
        for seed in tqdm(G.vs.indices): #for each node
            power_n = []
            for i in (range(num_iter)): #go over all worlds
                nodes_i = ICM(G,start_nodes=[seed]) 
                power_n.append(nodes_i) # get all cascades in each trial
            cascade.append(power_n) # add all cascades of a node to global cascade
        
        df_cascade[f'p={prob}'] = cascade

    df_cascade.to_csv(os.path.join("Cascade outputs",f"{dataset.split('.')[0]}_heuristics.csv"),index=False)
    df_prob.to_csv(f"{dataset.split('.')[0]}_weights.csv",index=False)