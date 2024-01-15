import pandas as pd
import numpy as np
import torch as t
import re
from alan.experiment_utils import seed_torch
import glob
import os

weather = pd.read_csv('2020Release_Nor/weather.csv')

weather = weather.loc[~weather['StartTemp'].astype(str).str.contains('NULL')]
weather = weather.loc[~weather['StartTemp'].isna()]

weather['StartTemp'] = weather['StartTemp'].astype(int)
routes = weather['RouteDataID'].sort_values().unique()
years = weather['Year'].sort_values().unique()

# weather.set_index(['Year', 'RouteDataID'], inplace=True)

all_files = glob.glob('2020Release_Nor/States/*.csv')

dfs = []

for file in all_files:
    df = pd.read_csv(file)
    df = df.loc[df['RouteDataID'].isin(routes) & df['Year'].isin(years)]
    dfs.append(df)

i=0
k = 0
while i < 10:
    print(i)
    seed_torch(k)
    k += 1






    try:
        birds = pd.concat(dfs, axis=0, ignore_index=True).reset_index()
        birds = birds.drop(columns='index')

        # birds.set_index(['Year', 'RouteDataID'], inplace=True)

        birds = birds[['RouteDataID', 'Year', 'AOU', 'Count10', 'Count20', 'Count30', 'Count40', 'Count50']]

        M = 6
        J = 12
        I = 100

        # print(birds[['Count10', 'Count20', 'Count30', 'Count40', 'Count50']].max)
        weather['RouteDataID'] = weather['RouteDataID'] + weather['Year']
        birds['RouteDataID'] = birds['RouteDataID'] + birds['Year'] #+ birds['AOU']
        df_new = birds.reset_index().copy()
        df_id = df_new.groupby(['Year', 'AOU']).apply(lambda x: len(x['RouteDataID'].unique())>I)
        aous = df_id[df_id].index.get_level_values('AOU').to_list()
        df_id = df_id[df_id]

        df_id = df_id.reset_index()
        df_boro = df_id.groupby('Year').apply(lambda x: len(x['AOU'].unique())>J)
        years = df_boro[df_boro].index.get_level_values('Year').to_list()
        df_boro = df_boro[df_boro]



        random_years = np.random.choice(years, M, replace=False)
        df_new = df_new.loc[df_new['Year'].isin(random_years)]

        new_aous = []
        for m in df_new['Year'].unique():
            new_aous.extend(np.random.choice(df_new.loc[df_new['Year'] == m].loc[df_new['AOU'].isin(aous) & ~df_new['AOU'].isin(new_aous)]['AOU'].unique(), J, replace=False))

        df_new = df_new.loc[df_new['AOU'].isin(new_aous)]
        num_aous = len(df_new['AOU'].unique())


        new_ids = []
        for j in df_new['AOU'].unique():

            new_ids.extend(np.random.choice(df_new.loc[df_new['AOU'] == j]['RouteDataID'].unique(), I, replace=False))

        df_new = df_new.loc[df_new['RouteDataID'].isin(new_ids)]


        df_new = df_new.merge(weather, on='RouteDataID')
        weather_new = df_new[['StartTemp']].to_numpy()[:df_new.shape[0]-df_new.shape[0]%(M*J*I),:].reshape(M,J,-1)[:,:,:300]
        quality = df_new[['QualityCurrentID']].to_numpy()[:df_new.shape[0]-df_new.shape[0]%(M*J*I),:].reshape(M,J,-1)[:,:,:300]

        df_new[['Count10', 'Count20', 'Count30', 'Count40', 'Count50']] = (df_new[['Count10', 'Count20', 'Count30', 'Count40', 'Count50']] > 0).astype(int)
        birds = df_new[['Count10', 'Count20', 'Count30', 'Count40', 'Count50']].to_numpy()[:df_new.shape[0]-df_new.shape[0]%(M*J*I),:].reshape(M,J,-1,5)[:,:,:300,:]


        t.save(t.from_numpy(weather_new)[:,:,:I//2], 'data/weather_train_{}.pt'.format(i))
        t.save(t.from_numpy(quality)[:,:,:I//2], 'data/quality_train_{}.pt'.format(i))
        t.save(t.from_numpy(birds)[:,:,:I//2,:], 'data/birds_train_{}.pt'.format(i))

        t.save(t.from_numpy(weather_new)[:,:,I//2:], 'data/weather_test_{}.pt'.format(i))
        t.save(t.from_numpy(quality)[:,:,I//2:], 'data/quality_test_{}.pt'.format(i))
        t.save(t.from_numpy(birds)[:,:,I//2:,:], 'data/birds_test_{}.pt'.format(i))

        i += 1
    except:
        None
