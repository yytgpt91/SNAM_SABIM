import os
# from statistics import mean
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
plt.rcParams.update({'font.size': 13})
plt.rc('font', weight='bold')
plt.rc("axes",titlesize = 22,labelsize = 22,labelweight = 'bold')
plt.rc("figure",titlesize = 22)
plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20) 

# plt.rc("axes",)

parser.add_argument("path",help="path to csv file") # link to dataset_name_merged.csv file
parser.add_argument("name",help="name of plot") # dataset name
args = parser.parse_args()

path_to_csv = args.path
dataset = args.name

df = pd.read_csv(path_to_csv)
methods = ['p=0.25','p=0.5','p=0.75','p=RA','p=AA','p=LHI','p=Jaccard','p=RA_decay','p=AA_decay','p=LHI_decay','p=Jaccard_decay']

mean_df = df.groupby("out_core")[methods].mean()
# print(mean_df)
std_df = df.groupby("out_core")[methods].std()

# figure, axis = plt.subplots(6, 2,sharex=True,sharey=True)

# for method,ax in zip(methods,axis.flat[:-1]):
#     ax.plot(sorted(df['out_core'].unique()),mean_df[method],marker = 'o',label = f"{method}",color = '#0000FF')
#     ax.plot(sorted(df['out_core'].unique()),mean_df[method]+std_df[method],color = '#808080')
#     ax.plot(sorted(df['out_core'].unique()),mean_df[method]-std_df[method],color = '#808080')

#     ax.fill_between(sorted(df['out_core'].unique()),
#                                     mean_df[method]+std_df[method],
#                                     mean_df[method]-std_df[method],
#                                     color = "#0000FF",alpha = 0.3)

#     # ax.xlabel('Shell Number')
#     # ax.ylabel('Cascading Power')
#     # ax.set(xlabel = '',ylabel = '')
#     # ax.set_title(f"Cascading power - {method}")
#     ax.legend()
# ax.show()

figure, ax = plt.subplots(figsize = (12,15))
# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(methods))))

name = "tab10"
cmap = plt.get_cmap(name)  #matplotlib.colors.ListedColormap
colors = cmap.colors 
colors = list(colors)
colors.append('blue')
# print(colors[0])

for c,method in zip(colors,methods):
    # c = next(color)
    ax.plot(sorted(df['out_core'].unique()),(mean_df[method]),marker = 'o',label = f"{method}",color = c)
    ax.plot(sorted(df['out_core'].unique()),(mean_df[method]+std_df[method]),color = '#808080',alpha = 0.5)
    ax.plot(sorted(df['out_core'].unique()),(mean_df[method]-std_df[method]),color = '#808080',alpha = 0.5)

    ax.fill_between(sorted(df['out_core'].unique()),
                                    (mean_df[method]+std_df[method]),
                                    (mean_df[method]-std_df[method]),
                                    color = c,alpha = 0.3)

    ax.legend()

ax.set(xlabel = 'Shell Number (Out)',ylabel = 'Cascading power ')
# ax.set_title(dataset+" Probability experiment",weight = 'bold')
ax.set_yscale("log")

if os.path.isdir('plots') == False:
    os.mkdir('plots')

plt.savefig(f"plots/{dataset}_cascade.png",dpi = 120)
plt.close()


#89CFF0