import torch as t
from alan import Normal, Plate, BoundPlate, Group, Problem, Data, mean, Split, OptParam, QEMParam, checkpoint, no_checkpoint, Split

device = 'cpu'                    #Options cpu, cuda, mps (on an Apple M)
computation_strategy = checkpoint #Options no_checkpoint, checkpoint, Split('p1', 3)

P_plate = Plate( 
    a = Normal(OptParam(0., name='a_loc_P'), 1),
    bc = Group(
        b = Normal('a', 1),
        c = Normal('b', 1),
    ),
    d = Normal(0, lambda c: c.exp()),
    p1 = Plate(
        e = Normal("d", 1),
        p2 = Plate(
            f = Normal("e", 1.),
        ),
    ),
)

Q_plate = Plate( 
    a = Normal(OptParam(0.), OptParam(1.)),
    bc = Group(
        b = Normal(QEMParam(0.), QEMParam(1.)),
        c = Normal('c_loc', lambda c_log_scale: c_log_scale.exp())
    ),
    d = Normal(0, lambda c: c.exp()),
    p1 = Plate(
        e = Normal(QEMParam(0.), QEMParam(1.)),
        p2 = Plate(
            f = Data(),
        ),
    ),
)

all_platesizes = {'p1': 4, 'p2': 6}
extra_opt_params = {'c_loc': t.zeros(()), 'c_log_scale': t.zeros(())}

P_bound_plate = BoundPlate(P_plate, all_platesizes)
Q_bound_plate = BoundPlate(Q_plate, all_platesizes, extra_opt_params=extra_opt_params)

P_sample = P_bound_plate.sample()
data = {'f': P_sample['f']}

problem = Problem(P_bound_plate, Q_bound_plate, data)
#Move problem to the device.
problem.to(device=device)

sample = problem.sample(K=10)



#Update QEM Params
sample.update_qem_params(0.1, computation_strategy=computation_strategy)



#Update Opt Params using VI
vi_opt = t.optim.Adam(problem.parameters(), lr=0.01, maximize=True)

sample.elbo_vi(computation_strategy=computation_strategy).backward()
vi_opt.step()
vi_opt.zero_grad()



#Update Opt Params using RWS.
rws_P_opt = t.optim.Adam(problem.P.parameters(), lr=0.01, maximize=True)
rws_Q_opt = t.optim.Adam(problem.Q.parameters(), lr=0.01, maximize=False)

sample.elbo_rws(computation_strategy=computation_strategy).backward()
rws_P_opt.step()
rws_Q_opt.step()

rws_P_opt.zero_grad()
rws_Q_opt.zero_grad()
