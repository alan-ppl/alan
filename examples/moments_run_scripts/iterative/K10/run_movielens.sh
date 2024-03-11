lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _MVIIS10   --queue cnu --cmd 'python runner_moments_iterative.py model=movielens K=10 method=vi'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _MVIIS10_F --queue cnu --cmd 'python runner_moments_iterative.py model=movielens K=10 method=vi fake_data=True'

lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _MRWSIS10   --queue cnu --cmd 'python runner_moments_iterative.py model=movielens K=10 lrs=[0.3,0.1,0.03,0.01] method=rws'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _MRWSIS10_F --queue cnu --cmd 'python runner_moments_iterative.py model=movielens K=10 lrs=[0.3,0.1,0.03,0.01] method=rws fake_data=True'
