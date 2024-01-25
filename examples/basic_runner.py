import torch as t
import importlib.util
import sys

Q_PARAM_TYPES = {'vi': 'opt', 'rws': 'opt', 'qem': 'qem'}

def run(model_name, 
        methods = ['vi', 'rws', 'qem'],
        K = 3, 
        num_runs = 1, 
        num_iters = 100, 
        lrs = {'vi': 0.1, 'rws': 0.1, 'qem': 0.1}, 
        reparam = True,
        do_plot = True, 
        do_predll = True, 
        N = 10,
        fake_data = False,
        dataset_seed = 0,
        data_dir = 'data/',
        device = 'cpu'):

    # ensure device is set correctly
    if device == 'cuda':
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print("Device:", device)

    # import the model
    spec = importlib.util.spec_from_file_location(model_name, f"{model_name}.py")
    model = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = model
    spec.loader.exec_module(model)

    # select the correct load_and_generate_problem function
    if fake_data:
        def load_and_generate_problem(device, Q_param_type, run=dataset_seed, data_dir=data_dir):
            return model._load_and_generate_problem(device, Q_param_type, run, data_dir, fake_data=True)
    else:
        def load_and_generate_problem(device, Q_param_type, run=dataset_seed, data_dir=data_dir):
            return model._load_and_generate_problem(device, Q_param_type, run, data_dir, fake_data=False)

    # set up results dicts
    elbos = {method: t.zeros((num_runs, num_iters)).to(device) for method in methods}
    lls   = {method: t.zeros((num_runs, num_iters)).to(device) for method in methods}

    # run the experiment
    for num_run in range(num_runs):
        print(f"Run {num_run}")
        print()

        for method in methods:
            print(method.upper())
            t.manual_seed(num_run)
            prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, Q_PARAM_TYPES[method])

            for key in all_data.keys():
                all_data[key] = all_data[key].to(device)
            for key in all_covariates.keys():
                all_covariates[key] = all_covariates[key].to(device)

            if method == 'vi':
                opt = t.optim.Adam(prob.Q.parameters(), lr=lrs[method])
            elif method == 'rws':
                opt = t.optim.Adam(prob.Q.parameters(), lr=lrs[method], maximize=True)
            
            for i in range(num_iters):
                if method in ['vi', 'rws']:
                    opt.zero_grad()

                sample = prob.sample(K, reparam)
                if method == 'vi':
                    elbo = sample.elbo_vi()
                elif method == 'rws':
                    elbo = sample.elbo_rws()
                elif method == 'qem':
                    elbo = sample.elbo_nograd()    

                elbos[method][num_run, i] = elbo.detach()

                if do_predll:
                    importance_sample = sample.importance_sample(N=N)
                    extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                    ll = extended_importance_sample.predictive_ll(all_data)
                    lls[method][num_run, i] = ll['obs']
                    print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")
                else:
                    print(f"Iter {i}. Elbo: {elbo:.3f}")

                if method in ['vi', 'rws']:
                    (-elbo).backward()
                    opt.step()
                else:
                    sample.update_qem_params(lrs[method])
        
            print()

    if do_plot:
        for key in elbos.keys():
            elbos[key] = elbos[key].cpu()
            lls[key] = lls[key].cpu()

        import matplotlib.pyplot as plt

        plt.figure()
        for method in methods:
            plt.plot(t.arange(num_iters), elbos[method].mean(0), label=f'{method.upper()} lr={lrs[method]}')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.title(f'{model_name.upper()} ({"FAKE" if fake_data else "REAL"} data)')
        plt.savefig('plots/quick_elbos.png')

        if do_predll:
            plt.figure()
            for method in methods:
                plt.plot(t.arange(num_iters), lls[method].mean(0), label=f'{method.upper()} lr={lrs[method]}')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.title(f'{model_name.upper()} ({"FAKE" if fake_data else "REAL"} data)')
            plt.savefig('plots/quick_predlls.png')
