lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 6 -m 64 -a ********** -n _MHMC   --queue cnu --cmd 'python runner_moments_HMC.py model=movielens'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 6 -m 64 -a ********** -n _MHMC_F --queue cnu --cmd 'python runner_moments_HMC.py model=movielens fake_data=True'
