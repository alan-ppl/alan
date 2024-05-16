import pandas as pd
import numpy as np
import torch as t
import re

i = 0
k = 0

def get_data():
    data = pd.read_csv('Bus_Breakdown_and_Delays.csv', header=0, encoding='latin-1')
    data = data.drop(['Schools_Serviced','Number_Of_Students_On_The_Bus','Has_Contractor_Notified_Schools','Has_Contractor_Notified_Parents','Have_You_Alerted_OPT','Informed_On','Incident_Number','Last_Updated_On', 'School_Age_or_PreK'], axis=1)
    data = data.loc[~data['How_Long_Delayed'].str.contains('/', na=False)]
    data['How_Long_Delayed'] = data['How_Long_Delayed'].map(lambda x: re.sub("[^0-9]", "", str(x)))
    data = data.loc[data['How_Long_Delayed'] != '']
    data['How_Long_Delayed'] = data['How_Long_Delayed'].map(lambda x: float(x))
    # print(len(data['How_Long_Delayed']))
    data['How_Long_Delayed'] = (data['How_Long_Delayed'] > 30).astype(float)
    
    # print(len(data['How_Long_Delayed']))
    data = data.dropna().reset_index(drop=True)
    return data

df = get_data()


while i < 5:
    print(i)
    t.manual_seed(k)
    k += 1





    M = 2
    J = 3
    I = 300



    try:
        df_new = df.reset_index().copy()
        df_id = df_new.groupby(['School_Year', 'Boro']).apply(lambda x: len(x['Busbreakdown_ID'].unique())>I)
        boros = df_id[df_id].index.get_level_values('Boro').to_list()
        df_id = df_id[df_id]

        df_id = df_id.reset_index()
        df_boro = df_id.groupby('School_Year').apply(lambda x: len(x['Boro'].unique())>J)
        years = df_boro[df_boro].index.get_level_values('School_Year').to_list()
        df_boro = df_boro[df_boro]



        random_years = np.random.choice(years, M, replace=False)
        df_new = df_new.loc[df_new['School_Year'].isin(random_years)]



        new_boros = []
        for m in df_new['School_Year'].unique():
            new_boros.extend(np.random.choice(df_new.loc[df_new['School_Year'] == m].loc[df_new['Boro'].isin(boros) & ~df_new['Boro'].isin(new_boros)]['Boro'].unique(), J, replace=False))

        df_new = df_new.loc[df_new['Boro'].isin(new_boros)]
        num_boros = len(df_new['Boro'].unique())


        new_ids = []
        for j in df_new['Boro'].unique():

            new_ids.extend(np.random.choice(df_new.loc[df_new['Boro'] == j]['Busbreakdown_ID'].unique(), I, replace=False))

        df_new = df_new.loc[df_new['Busbreakdown_ID'].isin(new_ids)]


        num_ids = len(df_new['Busbreakdown_ID'].unique())

        delay = df_new['How_Long_Delayed'].to_numpy()
        
        df_new.drop(columns='How_Long_Delayed', inplace=True)
        run_type = pd.get_dummies(df_new['Run_Type']).to_numpy()

        Bus_Company_Name = pd.get_dummies(df_new['Bus_Company_Name']).to_numpy()


        run_type = run_type.reshape(M, J, I, -1)

        bus_company_name = Bus_Company_Name.reshape(M, J, I, -1)

        delay = delay.reshape(M, J, I)
        print(delay)
        t.save(t.from_numpy(run_type)[:,:,:I//2,:], 'data/run_type_train_{}.pt'.format(i))
        t.save(t.from_numpy(bus_company_name)[:,:,:I//2,:], 'data/bus_company_name_train_{}.pt'.format(i))
        t.save(t.from_numpy(delay)[:,:,:I//2], 'data/delay_train_{}.pt'.format(i))

        t.save(t.from_numpy(run_type)[:,:,I//2:,:], 'data/run_type_test_{}.pt'.format(i))
        t.save(t.from_numpy(bus_company_name)[:,:,I//2:,:], 'data/bus_company_name_test_{}.pt'.format(i))
        t.save(t.from_numpy(delay)[:,:,I//2:], 'data/delay_test_{}.pt'.format(i))

        i += 1
    except:
        None