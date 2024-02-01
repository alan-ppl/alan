import pickle

def get_stats(results_location):
    print(f"results_location: {results_location}")
    for method in ['rws', 'qem', 'vi']:
        try:
            with open(f"{results_location}/{method}0.pkl", 'rb') as f:
                results = pickle.load(f)
            print(f"""method: {method}
        Ks: {results['Ks']}
        lrs: {results['lrs']} (actually elbos has {results['elbos'].shape[1]} lrs)
        num_iters: {results['num_iters']}
        num_runs: {results['num_runs']}\n""")

        except:
            print(f"Couldn't find {results_location}/{method}0.pkl")

        
if __name__ == "__main__":
    # get_stats('bus_breakdown/results/final1-0.0001')

    get_stats('chimpanzees/results/K5_15_lr_0.001-1')
    get_stats('chimpanzees/results/K10_lr_0.001-1')
    get_stats('chimpanzees/results/K5_10_15_lr_0.001-1')

    # get_stats('movielens/results/regular_version_final_FULL_all_lrs')

    # get_stats('occupancy/results/lr0.01-1')


    # get_stats('movielens/results/regular_version_final')
    # get_stats('movielens/results/regular_version_final_FULL')
    # get_stats('movielens/results/regular_version_final_FULL_all_lrs')
    # get_stats('movielens/results/0.2_and_0.3')

    # get_stats('radon/results')
    
