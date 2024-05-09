import torch as t
from alan import mean, mean2
from covid import generate_problem, load_data_covariates
import pickle

from torch.distributions import NegativeBinomial
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device, 0, 'data/')

prob = generate_problem(device, platesizes, data, covariates, Q_param_type='qem')

prob.load_state_dict(t.load('results/qem00.1.pth', map_location=torch.device('cpu') if device == 'cpu' else device))


K=10

sample = prob.sample(K)


names = ['CM_mean', 'CM_ex2', 'Wearing_mean', 'Wearing_ex2', 'Mobility_mean', 'Mobility_ex2']
#Moments of the weights
desired_moments = (('CM_alpha', mean), ('CM_alpha', mean2), ('Wearing_alpha', mean), ('Wearing_alpha', mean2), ('Mobility_alpha', mean), ('Mobility_alpha', mean2))
moments = dict(zip(names, sample.moments(desired_moments)))

#convert moments to numpy
for key in moments:
    moments[key] = moments[key].detach().cpu().numpy()
    
#save moments to file
with open('results/moments.pkl', 'wb') as f:
    pickle.dump(moments, f)
    
# predictive samples
names = ['log_infected', 'log_infected_ex2', 'Psi', 'Psi_ex2']
desired_moments = (('log_infected', mean), ('log_infected', mean2), ('Psi', mean), ('Psi', mean2))

moments = dict(zip(names, sample.moments(desired_moments)))

Psi = moments['Psi'].rename(None)
log_infected = moments['log_infected'].rename(None)

predicted_obs = {'obs':NegativeBinomial(total_count = t.exp(Psi), logits=t.exp(Psi)/(t.exp(Psi) + t.exp(log_infected) + 1e-4)).sample(t.Size([100]))}

print(predicted_obs)

#convert to numpy
predicted_obs['obs'] = predicted_obs['obs'].detach().cpu().numpy()

#save predictive samples to file
with open('results/predictive_samples.pkl', 'wb') as f:
    pickle.dump(predicted_obs, f)

