import pandas as pd
import numpy as np
import torch as t

from pathlib import Path
np.random.seed(0)

#states
S=4
#Zips
N=400



df_srrs2 = pd.read_csv('srrs2.dat')
df_cty = pd.read_csv('cty.dat')


df_srrs2.rename(columns=str.strip, inplace=True)
df_cty.rename(columns=str.strip, inplace=True)

df_srrs2['state'] = df_srrs2['state2'].str.strip()
df_srrs2['county'] = df_srrs2['county'].str.strip()

# We will now join datasets on Federal Information Processing Standards
# (FIPS) id, ie, codes that link geographic units, counties and county
# equivalents. http://jeffgill.org/Teaching/rpqm_9.pdf
df_srrs2['fips'] = 1000 * df_srrs2.stfips + df_srrs2.cntyfips
df_cty['fips'] = 1000 * df_cty.stfips + df_cty.ctfips

df = df_srrs2.merge(df_cty[['fips', 'Uppm', 'lon', 'lat']], on='fips')
df = df.drop_duplicates(subset='idnum')


df['wave'].replace({'  .': '-1'}, inplace=True)
df['rep'].replace({' .': '-1'}, inplace=True)
df['zip'].replace({'     ': '-1'}, inplace=True)
# Compute log(radon + 0.1)
df["log_radon"] = np.log(df["activity"] + 0.1)

# Compute log of Uranium
df["log_u"] = np.log(df["Uppm"]+0.1)


# Let's map floor. 0 -> Basement and 1 -> Floor
df['basement'] = df['basement'].apply(lambda x: 1 if x == 'Y' else 0)


df = df[['stfips', 'cntyfips', 'fips', 'log_u', 'basement', 'log_radon', 'zip']]
df['cntyfips'] = df['stfips'].map(str) + df['cntyfips'].map(str)
df['zip'] = df['cntyfips'] + df['zip']



#Select S states
states = df['stfips'].unique()
np.random.shuffle(states)
states = states[:S]
df = df[df['stfips'].isin(states)]


#Select N zips from each state
df = df.groupby(['stfips', 'zip'], as_index=False).first()
df = df.groupby(['stfips']).filter(lambda x: len(x['zip'].unique()) >= N)
df = df.groupby(['stfips']).head(N)


#reshaped pytorch tensors
basement = t.tensor(df['basement'].values).reshape(S,300).float()
log_u = t.tensor(df['log_u'].values).reshape(S,300).float()
log_radon = t.tensor(df['log_radon'].values).reshape(S,300).float()

#save
Path("data/").mkdir(parents=True, exist_ok=True)
t.save(basement, 'data/basement.pt')
t.save(log_u, 'data/log_u.pt')
t.save(log_radon, 'data/log_radon.pt')








# df = df.reset_index(drop=True)

# df_first_reading = df.groupby(['stfips', 'cntyfips', 'zip'], as_index=False).first()

# #Only keep states and counties at least N unique zips
# df_first_reading = df_first_reading.groupby(['stfips', 'cntyfips']).filter(lambda x: len(x['zip'].unique()) >= N)
# #select first N unique zips for each state and county
# df_first_reading = df_first_reading.groupby(['stfips', 'cntyfips']).head(N)

# #Only keep states and counties at least M unique zips
# df_first_reading = df_first_reading.groupby(['stfips']).filter(lambda x: len(x['cntyfips'].unique()) >= M)
# #Find the first M unique counties in each state
# unique_counties_for_each_state = (df_first_reading[['stfips', 'cntyfips']].drop_duplicates(subset=['stfips', 'cntyfips']).groupby(['stfips']).head(M))
# #Use this to filter the first readings
# df_first_reading = df_first_reading.merge(unique_counties_for_each_state, on=['stfips', 'cntyfips'])
# df_first_reading = df_first_reading[['stfips', 'cntyfips', 'zip', 'log_u', 'basement', 'log_radon']]


# print(df_first_reading)
# #Number of states
# S = len(df_first_reading['stfips'].unique())
# print(S)

# #reshaped pytorch tensors
# basement = t.tensor(df_first_reading['basement'].values).reshape(S,M,N).float()
# log_u = t.tensor(df_first_reading['log_u'].values).reshape(S,M,N).float()
# log_radon = t.tensor(df_first_reading['log_radon'].values).reshape(S,M,N).float()

# #Save these
# Path("data/").mkdir(parents=True, exist_ok=True)
# t.save(basement, 'data/basement.pt')
# t.save(log_u, 'data/log_u.pt')
# t.save(log_radon, 'data/log_radon.pt')








