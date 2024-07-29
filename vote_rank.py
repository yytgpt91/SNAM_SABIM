from matplotlib import pyplot as plt
import numpy as np
from random import random
from igraph import *
import os
import sys
import pandas as pd
import networkx as nx
import time
from tqdm import tqdm

sys.path.append("..")     
from utils import *

def get_seed_and_size(G,seed_size,no_iter):

       
    st_time = time.time()
    # ris_seeds = (G,seed_size,no_iter)
    A = G.get_edgelist()
    g = nx.DiGraph(A)

    seeds = nx.voterank(g,seed_size)
    
    print(time.time()-st_time)
    # print(ris_seeds)

    inf_size = 0
    inf_list = []
    for i in tqdm(range(no_iter)):
        inf_size += len(ICM(G,seeds))
        inf_list.append(inf_size/(i+1))
    
    # fig, ax = plt.subplots()

    # ax.plot([i+1 for i in range(no_iter)],inf_list)
    # plt.show()

    return seeds,inf_size/no_iter

df_results = pd.DataFrame(None,columns=["Dataset","Original","Degree",'RA','AA','LHI','Jaccard'])

for data in [("celegans_n306.txt","d","w"),("USairport500.txt","d","w"),
("Freemans_EIES-3_n32.txt","d","w"),("OClinks_w.txt","d","w"),("USairport_2010.txt","d","w")]:



# ("celegans_n306.txt","d","w"),("USairport500.txt","d","w"),
# ("Freemans_EIES-3_n32.txt","d","w"),

# ("USairport500.txt","d","w"),
# ("Freemans_EIES-3_n32.txt","d","w"),("OClinks_w.txt","d","w"),
# ("USairport_2010.txt","d","w")
   
    dataset,directed,weighted = data

    print(f"--------{dataset}---------")
    
    idx = len(df_results.index)
    
    df_results.at[idx,"Dataset"] = dataset
    
    seed_size = 2
    num_iter = 1000

    if dataset in ["email.txt" ,"road.txt" ,"weblog.txt" ,"animal_interaction.txt","facebook_combined.txt"]:
        data_file_path = os.path.join(os.pardir,os.pardir,"data","Prev Data",dataset)
    else:
        data_file_path = os.path.join(os.pardir,os.pardir,"data",dataset)
    
    directed = True if directed == "d" else False
    weighted = True if weighted == "w" else False

    G = read_graph(data_file_path,directed,weighted = weighted)

    G.es['p'] = scale(get_true_weights(dataset,directed,data_file_path,weighted))
    
    original_results = get_seed_and_size(G,seed_size,num_iter)

    df_results.at[idx,"Original"] = original_results
    print("Original ",original_results)

    

    # G.es['p'] = compute_prob(G)
    # degree_results = Greedy_ICM(G,seed_size,num_iter)
    # df_results.at[idx,"Degree"] = degree_results
                  
    # print("Degree ",degree_results)

    # p = ['RA','AA','LHI','Jaccard']


    # for heuristic in p:
    #     G.es['p'] = prob_heuristics(G,heuristic)
    #     heuristic_result = Greedy_ICM(G,seed_size,num_iter)
        
    #     df_results.at[idx,heuristic] = heuristic_result
        
    #     print(heuristic,heuristic_result)

    # print("-------------------")
    # seeds = [117,118]
    # inf_size = 0
    # inf_list = []
    # for i in tqdm(range(num_iter)):
    #     inf_size += len(ICM(G,seeds))
    #     inf_list.append(inf_size/(i+1))
    
    # fig, ax = plt.subplots()

    # ax.plot([i+1 for i in range(num_iter)],inf_list)
    # print("20: ",inf_list[19],"1000",inf_list[-1])
    # plt.show()


    df_results.to_csv(f"Voterank_op_{seed_size}.csv",index=False)