import pandas as pd
import numpy as np
import torch as t



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

for i in range(10):
    np.random.seed(i)
    #Films
    Ns = [5,20]
    #Users
    Ms = [300,450]



    for N in Ns:
        for M in Ms:
            x = get_features()
            users = np.random.choice(x.shape[0], M, replace=False)
            films = np.random.choice(x.shape[1], 2*N, replace=False)

            train_weights = x[np.ix_(users ,films[:N])]
            test_weights = x[np.ix_(users ,films[N:])]
            train_data = get_ratings()[np.ix_(users ,films[:N])]
            test_data = get_ratings()[np.ix_(users ,films[N:])]

            t.save(train_data, 'data/data_y_{0}_{1}_{2}.pt'.format(N, M,i))
            t.save(train_weights, 'data/weights_{0}_{1}_{2}.pt'.format(N, M,i))

            t.save(test_data, 'data/test_data_y_{0}_{1}_{2}.pt'.format(N, M,i))
            t.save(test_weights, 'data/test_weights_{0}_{1}_{2}.pt'.format(N, M,i))
