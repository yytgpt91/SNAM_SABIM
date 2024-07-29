import numpy as np
from random import random,choice
from igraph import *
import os
import sys
import pandas as pd
# import networkx as nx
import time
from tqdm import tqdm

sys.path.append("..")     
from utils import *

def h_index(G,node,mode = 'out'):
    '''
    Return h-index of a node
    '''

    sorted_neighbor_degrees = sorted(G.degree(G.neighbors(node,mode),mode),reverse=True)
    h = 0
    for i in range(1, len(sorted_neighbor_degrees)+1):
        if sorted_neighbor_degrees[i-1] < i:
            break
        h = i

    return h


def hi_index(G,node,mode='out'):
    '''
    Return hi_index (h-index on h-index) of a node
    '''

    sorted_neighbor_h_index = sorted([h_index(G,v,mode) for v in G.neighbors(node,mode)],reverse=True)
    h = 0
    for i in range(1, len(sorted_neighbor_h_index)+1):
        if sorted_neighbor_h_index[i-1] < i:
            break
        h = i

    return h

# %%
def get_cost(G,source_node):
    '''
    Assign costs for each edge
    source_node : igraph Vertex
    '''
    
    # distance = Graph.get_shortest_paths(source_node,[v for v in G.vs if v!=source_node],mode='out',output='vpath')
    G.vs['h2-idx'] = [hi_index(G,node,'out') for node in G.vs]

    for node in G.vs:
        if node!=source_node:
            
            dist = len(G.get_shortest_paths(source_node,node,mode='out',output='vpath')[0])  #assign distance
            
            if dist == 0: #handle the case
                dist = np.inf
            
            node['dist'] = dist
            
            node['cost'] = np.exp(node['dist']) * (node['h2-idx'] - source_node['h2-idx']+ max(G.vs['h2-idx'])) #check

# %%
def get_inf_size(seeds_list):

    inf_size = 0
    inf_list = []
    for seeds in (seeds_list):
        inf_size += len(ICM(G,seeds))
        inf_list.append(inf_size/(i+1))
    
    # fig, ax = plt.subplots()

    # ax.plot([i+1 for i in range(no_iter)],inf_list)
    # plt.show()

    return inf_size/len(seeds_list)

def get_inf(seeds,no_iterations):

    inf_size = 0
    inf_list = []
    for i in range(no_iterations):
        inf_size += len(ICM(G,seeds))
        inf_list.append(inf_size/(i+1))
    
    # fig, ax = plt.subplots()

    # ax.plot([i+1 for i in range(no_iter)],inf_list)
    # plt.show()

    return inf_size/no_iterations

def GSM(G):

    for source_node in G.vs:

        inf = 0
        for node in G.vs:
            if node!= source_node:

                dist = len(G.get_shortest_paths(source_node,node,mode='out',output='vpath')[0])
                if dist == 0: #handle the case
                    dist = np.inf

                inf += node['shell']/dist
            
        source_node['GSM'] = np.exp(source_node['shell']/len(G.vs)) *inf 


datasets = [("celegans_n306.txt","d","w"),("USairport500.txt","d","w"),
("Freemans_EIES-3_n32.txt","d","w"),("OClinks_w.txt","d","w"),
("USairport_2010.txt","d","w")]

# ("celegans_n306.txt","d","w"),("USairport500.txt","d","w"),
# ("Freemans_EIES-3_n32.txt","d","w"),("OClinks_w.txt","d","w"),
# ("USairport_2010.txt","d","w"),

# ("lkml.txt","d","w"),
# ("soc-redditHyperlinks-body.txt","d","w"),("soc-redditHyperlinks-title.txt","d","w")

#("jazz.txt","u","w"),("petster-friendships-hamster-uniq.txt","u","u")
#("arenas-email.txt","u","u"), ("ia-crime-moreno.txt","u","u")
#("dolphins.txt","u","u") , ("web-spam.txt","u","u")
#("ca-AstroPh.txt","u","u")


seed_size = 10
no_iterations = 100

def GSM_neighbour(G,source_node):

    GSM(G)

    get_cost(G,source_node)

    for node in G.vs:
        if node!=source_node:

            node['GSM_neigh'] = node['GSM']/node['cost']
    
results = {}
methods = ['GSM_neigh']


for dataset,directed,weighted in (datasets):
    
    print(dataset)
    df_result = pd.DataFrame(None,columns=['seed_size','shell']+methods)

    if dataset in ["email.txt" ,"road.txt" ,"weblog.txt" ,"animal_interaction.txt","facebook_combined.txt"]:
        data_file_path = os.path.join(os.pardir,os.pardir,"data","Prev Data",dataset)
    else:
        data_file_path = os.path.join(os.pardir,os.pardir,"data",dataset)
    
    directed = True if directed == "d" else False
    weighted = True if weighted == "w" else False

    G = read_graph(data_file_path,directed,weighted = weighted)
    G.es['p'] = scale(get_true_weights(dataset,directed,data_file_path,weighted))

    # df = pd.read_csv(os.path.join(os.pardir,"Cascade_Experiments","Cascade outputs",f"{dataset.split('.')[0]}_CP.csv"))

    G.vs['shell'] = G.coreness(mode='out')

    
    # seeds_list = []
    G.vs['h2-idx'] = [hi_index(G,node,'out') for node in G.vs]

    # new_heuristic(G)
    # GSM(G)
    # neigh_heur(G)
    
    
    
    # source_node = choice(G.vs)
    # get_cost(G,source_node)

    for index in (np.unique(G.vs['shell'])):
        print(f"--{index}--")
        for method in methods:
        
            nodes_per_index = G.vs.select(lambda vertex:vertex['shell'] == index)
            # source_node = max(nodes_per_index,key = lambda node: node.outdegree()) 
            source_node = min(nodes_per_index,key = lambda node: node.outdegree()) 


            GSM_neighbour(G,source_node)
        
            seeds = []
            for s in tqdm(range(seed_size)):
                idx = len(df_result.index)
                df_result.at[idx,'seed_size'] = s+1
                df_result.at[idx,'shell'] = index
                seed = max([node for node in G.vs if node not in seeds+[source_node]],key= lambda node:node[method])
                seeds.append(seed)
                df_result.at[idx,method] =  ([seed.index for seed in seeds],get_inf(seeds,no_iterations))

        # seeds_list.append(seeds)
    
            results[dataset] = df_result
            results[dataset].to_csv(os.path.join("local_GSM",f"{dataset.split('.')[0]}_GSM_neigh.csv"),index=False)
            
        # print(dataset,get_inf_size(seeds_list))


# %%
# for dataset in results.keys():
