import os

import numpy as np
import torch as t
np.random.seed(123456)

import sys
import argparse
import datetime
import pickle
import alan
# import pymc3 as pm

from models.epi_params import EpidemiologicalParameters
from models.preprocessing.preprocess_mask_data import Preprocess_masks
from models.mask_models_weekly import P_plate, Q_plate, model_data

from alan import BoundPlate, Problem


argparser = argparse.ArgumentParser()
argparser.add_argument("--model", dest="model", type=str, help="Model type")
argparser.add_argument("--masks", dest="masks", type=str, help="Which mask feature")
argparser.add_argument(
    "--w_par", dest="w_par", type=str, help="Which wearing parameterisation"
)
argparser.add_argument("--mob", dest="mob", type=str, help="How to include mobility")

# argparser.add_argument('--filter', dest='filtered', type=str, help='How to remove regions')
# argparser.add_argument('--gatherings', dest='gatherings', type=int, help='how many gatherings features')
argparser.add_argument("--ML", dest="ml", type=bool, help="Whether to run ML update")
# argparser.add_argument('--hide_ends', dest='hide_ends', type=str)
args, _ = argparser.parse_known_args()

#MODEL = args.model
MODEL = 'cases'
# MASKS = args.masks
MASKS = 'wearing'
W_PAR = args.w_par if args.w_par else "exp"
# MOBI = args.mob
MOBI='include'
# ML = args.ml
ml = True
# FILTERED = args.filtered

US = True
SMOOTH = False
GATHERINGS = 3  # args.gatherings if args.gatherings else 3
# MASKING = True # Always true


# prep data object
path = f"models/data/modelling_set/master_data_mob_{MOBI}_us_{US}_m_w.csv"

masks_object = Preprocess_masks(path)
masks_object.featurize(gatherings=GATHERINGS, masks=MASKS, smooth=SMOOTH, mobility=MOBI)
masks_object.make_preprocessed_object()
data = masks_object.data

all_observed_active, nRs, nDs, nCMs = model_data(masks_object.data)



ActiveCMs = np.add.reduceat(masks_object.data.ActiveCMs, np.arange(0, nDs, 7), 2)


ActiveCMs = t.from_numpy(np.moveaxis(ActiveCMs,[0,2,1], [0,1,2]))

CMs = masks_object.data.CMs

#Number of weeks
nWs = int(np.ceil(nDs/7))


print('nRs')
print(nRs)

print('nDs')
print(nDs)

print('nCMs')
print(nCMs)

print('nWs')
print(nWs)

# model specification
ep = EpidemiologicalParameters()
bd = ep.get_model_build_dict()



def set_init_infections(data, d):
    n_masked_days = 10
    first_day_new = data.NewCases[:, n_masked_days]
    first_day_new = first_day_new[first_day_new.mask == False]
    median_init_size = np.median(first_day_new)


    if median_init_size == 0:
        median_init_size = 50

    return np.log(median_init_size), np.log(median_init_size)


log_init_mean, log_init_sd = set_init_infections(data, bd)

bd["wearing_parameterisation"] = W_PAR



r_walk_period = 7
nNP = int(nDs / r_walk_period) - 1


platesizes = {'nRs':nRs,
               'nWs':int(nWs*0.8)}

all_platesizes = {'nRs':nRs,
                'nWs':nWs}
#New weekly cases
newcases_weekly = np.nan_to_num(np.add.reduceat(data.NewCases, np.arange(0, nDs, 7), 1))
newcases_weekly = t.from_numpy(newcases_weekly).rename('nRs', 'nWs' )
#NPI active CMs
ActiveCMs = ActiveCMs.float()
ActiveCMs_NPIs = ActiveCMs[:, :, :-2].rename('nRs', 'nWs', None)

ActiveCMs_wearing = ActiveCMs[:, :, -1].rename('nRs', 'nWs' )
ActiveCMs_mobility = ActiveCMs[:, :, -2].rename('nRs', 'nWs')

covariates = {'ActiveCMs_NPIs':ActiveCMs_NPIs[:,:platesizes['nWs'],:], 'ActiveCMs_wearing':ActiveCMs_wearing[:,:platesizes['nWs']], 'ActiveCMs_mobility':ActiveCMs_mobility[:,:platesizes['nWs']]}
all_covariates = {'ActiveCMs_NPIs':ActiveCMs_NPIs, 'ActiveCMs_wearing':ActiveCMs_wearing, 'ActiveCMs_mobility':ActiveCMs_mobility}

data = {'obs':newcases_weekly[:,:platesizes['nWs']]}
all_data = {'obs':newcases_weekly}



P_bound_plate = BoundPlate(P_plate, platesizes, inputs=covariates)
Q_bound_plate = BoundPlate(Q_plate, platesizes, inputs=covariates)

prob = Problem(P_bound_plate, Q_bound_plate, data)


opt = t.optim.Adam(prob.Q.parameters(), lr=0.01)
K=3

for i in range(100):
    opt.zero_grad()

    sample = prob.sample(K, True)
    elbo = sample.elbo_vi()

    if DO_PREDLL:
        importance_sample = sample.importance_sample(N=10)
        extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
        ll = extended_importance_sample.predictive_ll(all_data)
        print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")
    else:
        print(f"Iter {i}. Elbo: {elbo:.3f}")

    (-elbo).backward()
    opt.step()