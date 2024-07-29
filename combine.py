import ast
from numpy import mean, std
import pandas as pd
from functools import reduce
import os
from tqdm import tqdm

datasets = [("facebook_combined.txt","u")]
# ("zachary.graphml","u"),("email.txt","u") ,("road.txt","u") ,("weblog.txt","u")
#       ,("animal_interaction.txt","u")
for dataset,_ in tqdm(datasets):
    dataset = dataset.split(".")[0]
    df_p = pd.read_csv(os.path.join("Cascade outputs",f'{dataset}_constp.csv'))
    df_h = pd.read_csv(os.path.join("Cascade outputs",f'{dataset}_heuristics.csv'))
    df_hd = pd.read_csv(os.path.join("Cascade outputs",f'{dataset}_heuristics_decay_root_t.csv'))



    data_frames = [df_p,df_h,df_hd]

    # df_merged = pd.merge(df_p,df_h,on=['Node','out_core'])
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Node','out_core'],
                                                how='inner'), data_frames)
    df_merged.to_csv(os.path.join("Cascade outputs",f"{dataset}_merged.csv"),index= False)

    df_cp = pd.DataFrame(None)
    df_cp['Node'] = df_merged['Node']
    df_cp['out_core'] = df_merged['out_core']

    for column in df_merged.columns[2:]:
        df_cp[column] = df_merged[column].apply(lambda x:mean([len(l) for l in (ast.literal_eval(x))]) )
        # df_cp[column+"_std"] = df_merged[column].apply(lambda x:std([len(l) for l in (ast.literal_eval(x))]) )

    df_cp.to_csv(os.path.join("Cascade outputs",f"{dataset}_CP.csv"),index = False)