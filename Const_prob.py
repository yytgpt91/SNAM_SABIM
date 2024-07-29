from random import random
from igraph import *
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from utils import *

#-------------------------main--------------------

for data in  [("email.txt","u") ,("road.txt","u") ,("weblog.txt","u")
      ,("animal_interaction.txt","u"),("zachary.graphml","u") ]:
    #("facebook_combined.txt","u"),("zachary.graphml","u")
    # ("email.txt","u") ,("road.txt","u") ,("weblog.txt","u")
    #  ,("animal_interaction.txt","u")
    
    dataset,directed = data
    if dataset in ["email.txt" ,"road.txt" ,"weblog.txt" ,"animal_interaction.txt","facebook_combined.txt"]:
        data_file_path = os.path.join(os.pardir,os.pardir,"data","Prev Data",dataset)
    else:
        data_file_path = os.path.join(os.pardir,os.pardir,"data",dataset)
    
    directed = True if directed == "d" else False
    G = read_graph(data_file_path,directed)

    num_iter = 20
    p = [0.25,0.5,0.75]

    df_cascade = pd.DataFrame(None)
    df_cascade['Node'] = G.vs.indices
    df_cascade['out_core'] = G.coreness(mode='out')

    for prob in p:
        G.es['p'] = [prob]*G.ecount()
        # print(G.es['p'])
        # worlds = []
        # for i in range(num_iter):
        #     worlds.append(G.subgraph_edges([edge for edge in G.es if random() <= edge['p']],delete_vertices = False))
        
        cascade = []
        
        for seed in tqdm(G.vs.indices): #for each node
            power_n = []
            for i in (range(num_iter)): #go over all worlds
                nodes_i = ICM(G,start_nodes=[seed]) 
                power_n.append(nodes_i) # get all cascades in each trial
            cascade.append(power_n) # add all cascades of a node to global cascade
        
        df_cascade[f'p={prob}'] = cascade

    df_cascade.to_csv(os.path.join("Cascade outputs",f"{dataset.split('.')[0]}_constp.csv"),index=False)
    # print(G.ecount(),G.vcount())