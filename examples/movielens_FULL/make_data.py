import pandas as pd
import numpy as np
import torch as t
import alan

# from alan.experiment_utils import seed_torch
# from movielens import generate_model

def get_features():
    feats = pd.read_csv('ml-100k/u.item', sep='|', index_col=0, header=None, encoding='latin-1')
    feats = feats.drop([1,2,3,4,5], axis=1)
    feats.columns = ['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    # print(feats.head(5))
    feats = feats.to_numpy().repeat(943)
    feats = feats.reshape(943,1682,18)

    return t.tensor(feats).type(t.FloatTensor)


def get_ratings():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', header=None)
    ratings.columns = ['user id', 'item id',  'rating',  'timestamp']
    ratings['rating'].loc[ratings['rating'] < 4] = 0
    ratings['rating'].loc[ratings['rating'] >= 4] = 1
    ratings = ratings.pivot(index='user id', columns='item id', values='rating').fillna(0)
    # print(ratings.head(5))
    return t.tensor(ratings.to_numpy())

for i in range(1):
    t.manual_seed(i)

    x = get_features()

    M = x.shape[0]       # number of users (943)
    N = x.shape[1] // 2  # number of films (1682 // 2 = 841)

    films = np.random.choice(x.shape[1], 2*N, replace=False)

    train_weights = x[:,films[:N],:]
    test_weights = x[:,films[N:],:]

    train_data = get_ratings()[:,films[:N]]
    test_data = get_ratings()[:,films[N:]]

    t.save(train_data, f'data/data_y_{N}_{M}_{i}.pt')
    t.save(train_weights, f'data/weights_{N}_{M}_{i}.pt')

    t.save(test_data, f'data/test_data_y_{N}_{M}_{i}.pt')
    t.save(test_weights, f'data/test_weights_{N}_{M}_{i}.pt')
