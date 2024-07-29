import numpy as np
from random import random,choice
from igraph import *
import os
import sys
import pandas as pd
# import networkx as nx
import time
from tqdm import tqdm

sys.path.append("..")     # for getting utils.py which is located in parent folder
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
            # print(dist)
            
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

# (dataset name; d - directed, u - undirected; w - weighted, u - unweighted)
datasets = [("lkml.txt","d","w")]

# ("celegans_n306.txt","d","w"),("USairport500.txt","d","w"),
# ("Freemans_EIES-3_n32.txt","d","w"),("OClinks_w.txt","d","w"),
# ("USairport_2010.txt","d","w"),("jazz.txt","u","u"),("petster-friendships-hamster-uniq.txt","u","u"),
# ("arenas-email.txt","u","u"),("ia-crime-moreno.txt","u","u"),
# ("dolphins.txt","u","u"),("email.txt","u","u"),("weblog.txt","u","u") ,
# ("animal_interaction.txt","u","u")

("lkml.txt","d","w")

# 
# "facebook_combined.txt"

# ("lkml.txt","d","w"),
# ("soc-redditHyperlinks-body.txt","d","w"),("soc-redditHyperlinks-title.txt","d","w"),

# ("web-spam.txt","u","u"),("ca-AstroPh.txt","u","u")


#("jazz.txt","u","u"),("petster-friendships-hamster-uniq.txt","u","u")
#("arenas-email.txt","u","u"), ("ia-crime-moreno.txt","u","u")
#("dolphins.txt","u","u") , 

# ("jazz.txt","u","u"),("petster-friendships-hamster-uniq.txt","u","u"),
# ("arenas-email.txt","u","u"),("ia-crime-moreno.txt","u","u"),
# ("dolphins.txt","u","u")("web-spam.txt","u","u"),
# ("ca-AstroPh.txt","u","u")




def GSM_outdegree(G,source_node):

    for node in G.vs:
        if node != source_node:
            dist = len(G.get_shortest_paths(source_node,node,mode='out',output='vpath')[0])  #assign distance
            
            if dist == 0: #handle the case
                dist = np.inf
            
            node['cost'] = (node.outdegree()+1) * np.exp(dist)
            node['GSM_outdegree'] = node['GSM']/node['cost']

def GSM_outdegree_scaled(G,source_node):

    for node in G.vs:
        if node !=source_node:
            
            dist = len(G.get_shortest_paths(source_node,node,mode='out',output='vpath')[0])  #assign distance
            if dist == 0: #handle the case
                dist = np.inf
            node['cost'] = (node.outdegree()*G.vcount()+1)/sum(G.degree(mode='out')) *  (np.exp(dist))
            node['GSM_outdegree_scaled'] = node['GSM']/node['cost']


def GSM_h2(G,source_node):

    get_cost(G,source_node)

    for node in G.vs:
        if node!=source_node:

            node['GSM_h2'] = node['GSM']/node['cost']

def set_method(method:str,G,source_node):

    if method == 'GSM_h2':
        GSM_h2(G,source_node)
    elif method == 'GSM_outdegree':
        GSM_outdegree(G,source_node)
    elif method == "GSM_outdegree_scaled":
        GSM_outdegree_scaled(G,source_node)

results = {}
methods = ['GSM_h2','GSM_outdegree','GSM_outdegree_scaled']

#parameters 
seed_size = 5
no_iterations = 500
budget_factor = 0.5


for dataset,directed,weighted in (datasets):

    
    df_result = pd.DataFrame(None,columns=['seed_size','shell']+methods)

    if dataset in ["email.txt" ,"road.txt" ,"weblog.txt" ,"animal_interaction.txt","facebook_combined.txt"]:
        data_file_path = os.path.join(os.pardir,os.pardir,"data","Prev Data",dataset)
    else:
        data_file_path = os.path.join(os.pardir,os.pardir,"data",dataset)
    
    directed = True if directed == "d" else False
    weighted = True if weighted == "w" else False

    G = read_graph(data_file_path,directed,weighted = weighted)
    
    if weighted == True:
        G.es['p'] = scale(get_true_weights(dataset,directed,data_file_path,weighted)) #if weighted, get true weights scaled b/w 0 and 1
    else:
        G.es['p'] = prob_heuristics(G,"RA") # else use Resource allocation to assign weights

    # df = pd.read_csv(os.path.join(os.pardir,"Cascade_Experiments","Cascade outputs",f"{dataset.split('.')[0]}_CP.csv"))

    G.vs['shell'] = G.coreness(mode='out')

    
    # seeds_list = []
    G.vs['h2-idx'] = [hi_index(G,node,'out') for node in G.vs]

    # new_heuristic(G)
    # GSM(G)
    # neigh_heur(G)
    
    
    
    # source_node = choice(G.vs)
    # get_cost(G,source_node)    
    print(dataset, np.unique(G.vs['shell']))
    GSM(G)


    for i,index in enumerate(np.unique(G.vs['shell'])):
        
        print(f"--{index}--")
        for method in methods:
            
            nodes_per_index = G.vs.select(lambda vertex:vertex['shell'] == index)
            # source_node = max(nodes_per_index,key = lambda node: node.outdegree()) 
            for n in sorted(nodes_per_index,key = lambda node: node.outdegree()):
                if n.outdegree()!=0:
                    source_node = n
                    break

            # GSM_neighbour(G,source_node) 
            # call appropriate cost fn
            set_method(method,G,source_node)
            # print(G.vs['cost'])
            budget = max([node['cost'] for node in G.vs if (node!=source_node) and (node['cost']!=np.inf)]) * budget_factor
        
            seeds = []
            cost = 0
            for s in tqdm(range(seed_size)):
                idx = (i*10)+seed_size
                # idx = len(df_result.index)
                df_result.at[idx,'seed_size'] = s+1
                df_result.at[idx,'shell'] = index
                candidate_list = sorted([node for node in G.vs if node not in seeds+[source_node]],key= lambda node:node[method],reverse=True)
                for node in candidate_list:
                    if cost+node['cost'] <=budget:
                        seeds.append(node)
                        cost = cost + node['cost']
                        break
                # seeds.append(seed)
            df_result.at[idx,method] =  ([seed.index for seed in seeds],get_inf(seeds,no_iterations),cost,budget)

        # seeds_list.append(seeds)
    
    results[dataset] = df_result
    if os.path.exists('BIM_GSM') == False:
        os.mkdir("BIM_GSM")
    results[dataset].to_csv(os.path.join("BIM_GSM",f"{dataset.split('.')[0]}_B_GSM{seed_size}_{budget_factor}.csv"),index=False)
            
        # print(dataset,get_inf_size(seeds_list))


# %%
# for dataset in results.keys():
