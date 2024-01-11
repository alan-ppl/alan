#Bayesian logistic regression multi-class classification on MNIST
import torch as t
from alan import Normal, Categorical, Plate, BoundPlate, Problem, Data, mean, OptParam, QEMParam, checkpoint, no_checkpoint, Split
import torchvision
from pathlib import Path

def load_data_covariates(device, run, data_dir="data"):
    #Get MNIST from torch


    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False)


    #Cast to float 
    trainset.data = trainset.data.float()
    testset.data = testset.data.float()
    
    platesizes = {'plate1': trainset.data.shape[0]}
    all_platesizes = {'plate1': trainset.data.shape[0] + testset.data.shape[0]}

    train_data = {'y': trainset.targets.rename('plate1')}
    train_inputs = {'x': trainset.data.rename('plate1', ...)}

    all_data = {'y': t.cat([trainset.targets, testset.targets], dim=0).rename('plate1')}
    all_inputs = {'x': t.cat([trainset.data, testset.data], dim=0).rename('plate1', ...)}

    return platesizes, all_platesizes,  train_data, all_data, train_inputs, all_inputs

def generate_problem(device, platesizes, data, covariates, Q_param_type):
    
    P_plate = Plate(
        w = Normal(0, 1, sample_shape=t.Size([28**2, 10])),
        plate1 = Plate(
            y = Categorical(logits= lambda w, x: x.flatten() @ w),
        ),
    )

    if Q_param_type == "opt": 
        Q_plate = Plate(
            w = Normal(OptParam(0.), OptParam(1., transformation=t.exp), sample_shape=t.Size([28**2, 10])),
            plate1 = Plate(
                y = Data()
            ),
        )
    elif Q_param_type == "qem":
        Q_plate = Plate(
            w = Normal(QEMParam(t.zeros((28**2, 10))), QEMParam(t.ones((28**2, 10)))),
            plate1 = Plate(
                y = Data()
            ),
        )

    P_bound_plate = BoundPlate(P_plate, platesizes, inputs=covariates)
    Q_bound_plate = BoundPlate(Q_plate, platesizes)

    prob = Problem(P_bound_plate, Q_bound_plate, data)
    prob.to(device)

    return prob



def load_and_generate_problem(device, Q_param_type, run=0, data_dir='data/'):
    platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device, run, data_dir)
    problem = generate_problem(device, platesizes, data, covariates, Q_param_type)
    
    return problem, all_data, all_covariates, all_platesizes

if __name__ == "__main__":
    computation_strategy = Split('plate1', 32)
    Path("plots/MNIST_classification").mkdir(parents=True, exist_ok=True)
    DO_PLOT   = True
    DO_PREDLL = True
    NUM_ITERS = 100
    NUM_RUNS  = 1

    K = 10

    vi_lr = 0.1
    rws_lr = 0.1
    qem_lr = 0.1

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    # device='cpu'

    elbos = {'vi' : t.zeros((NUM_RUNS, NUM_ITERS)).to(device),
             'rws': t.zeros((NUM_RUNS, NUM_ITERS)).to(device),
             'qem': t.zeros((NUM_RUNS, NUM_ITERS)).to(device)}
    
    lls   = {'vi' : t.zeros((NUM_RUNS, NUM_ITERS)).to(device),
             'rws': t.zeros((NUM_RUNS, NUM_ITERS)).to(device),
             'qem': t.zeros((NUM_RUNS, NUM_ITERS)).to(device)}

    print(f"Device: {device}")

    for num_run in range(NUM_RUNS):
        print(f"Run {num_run}")
        print()
        print(f"VI")
        t.manual_seed(num_run)
        prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, 'opt')

        for key in all_data.keys():
            all_data[key] = all_data[key].to(device)
        for key in all_covariates.keys():
            all_covariates[key] = all_covariates[key].to(device)

        opt = t.optim.Adam(prob.Q.parameters(), lr=vi_lr)

        for i in range(NUM_ITERS):
            opt.zero_grad()

            sample = prob.sample(K, True)
            elbo = sample.elbo_vi(computation_strategy=computation_strategy)
            elbos['vi'][num_run, i] = elbo.detach()

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10, computation_strategy=computation_strategy)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['vi'][num_run, i] = ll['y']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['y']:.3f}")
            else:
                print(f"Iter {i}. Elbo: {elbo:.3f}")

            (-elbo).backward()
            opt.step()

        print()
        print(f"RWS")
        t.manual_seed(num_run)

        prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, 'opt')

        for key in all_data.keys():
            all_data[key] = all_data[key].to(device)
        for key in all_covariates.keys():
            all_covariates[key] = all_covariates[key].to(device)

        opt = t.optim.Adam(prob.Q.parameters(), lr=rws_lr)
        for i in range(NUM_ITERS):
            opt.zero_grad()

            sample = prob.sample(K, True)
            elbo = sample.elbo_rws(computation_strategy=computation_strategy)
            elbos['rws'][num_run, i] = elbo.detach()

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10, computation_strategy=computation_strategy)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['rws'][num_run, i] = ll['y']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['y']:.3f}")
            else:
                print(f"Iter {i}. Elbo: {elbo:.3f}")

            (-elbo).backward()
            opt.step()

        print()
        print(f"QEM")
        t.manual_seed(num_run)

        prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, 'qem')

        for key in all_data.keys():
            all_data[key] = all_data[key].to(device)
        for key in all_covariates.keys():
            all_covariates[key] = all_covariates[key].to(device)

        for i in range(NUM_ITERS):
            sample = prob.sample(K, True)
            elbo = sample.elbo_nograd(computation_strategy=computation_strategy)
            elbos['qem'][num_run, i] = elbo

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10, computation_strategy=computation_strategy)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['qem'][num_run, i] = ll['y']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['y']:.3f}")
            else:
                print(f"Iter {i}. Elbo: {elbo:.3f}")

            sample.update_qem_params(qem_lr, computation_strategy=computation_strategy)

    if DO_PLOT:
        for key in elbos.keys():
            elbos[key] = elbos[key].cpu()
            lls[key] = lls[key].cpu()

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(t.arange(NUM_ITERS), elbos['vi'].mean(0), label=f'VI lr={vi_lr}')
        plt.plot(t.arange(NUM_ITERS), elbos['rws'].mean(0), label=f'RWS lr={rws_lr}')
        plt.plot(t.arange(NUM_ITERS), elbos['qem'].mean(0), label=f'QEM lr={qem_lr}')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.title(f'MNIST_classification (K={K})')
        plt.tight_layout()
        plt.savefig('plots/MNIST_classification/quick_elbos.png')

        if DO_PREDLL:
            plt.figure()
            plt.plot(t.arange(NUM_ITERS), lls['vi'].mean(0), label=f'VI lr={vi_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['rws'].mean(0), label=f'RWS lr={rws_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['qem'].mean(0), label=f'QEM lr={qem_lr}')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.title(f'MNIST_classification (K={K})')
            plt.tight_layout()
            plt.savefig('plots/MNIST_classification/quick_predlls.png')