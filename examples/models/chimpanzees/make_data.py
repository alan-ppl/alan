import pandas as pd
import numpy as np
import torch as t
import alan
import pandas as pd

data = pd.read_csv('data/chimpanzees.csv')

num_actors  = len(np.unique(data['actor']))
num_blocks  = len(np.unique(data['block']))
num_repeats_per_condition = 6
num_repeats = num_repeats_per_condition *2

pulled_left = data['pulled_left'].to_numpy().reshape((num_actors,num_blocks,num_repeats))
prosoc_left = data['prosoc_left'].to_numpy().reshape((num_actors,num_blocks,num_repeats))
condition_ = data['condition']  .to_numpy().reshape((num_actors,num_blocks,num_repeats))

pulled_left = t.tensor(pulled_left).type(t.FloatTensor)
prosoc_left = t.tensor(prosoc_left).type(t.FloatTensor)
condition_ = t.tensor(condition_).type(t.FloatTensor)

pulled_left_train = pulled_left[:,:,:10]
pulled_left_test  = pulled_left[:,:,10:]
prosoc_left_train = prosoc_left[:,:,:10]
prosoc_left_test  = prosoc_left[:,:,10:]
condition_train   = condition_[:,:,:10]
condition_test    = condition_[:,:,10:]

t.save(pulled_left_train, 'data/data_train.pt')
t.save(pulled_left_test, 'data/data_test.pt')
t.save(prosoc_left_train, 'data/prosoc_left_train.pt')
t.save(prosoc_left_test, 'data/prosoc_left_test.pt')
t.save(condition_train, 'data/condition_train.pt')
t.save(condition_test, 'data/condition_test.pt')
