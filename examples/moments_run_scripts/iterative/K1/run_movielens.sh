lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _MVIIS   --queue cnu --cmd 'python runner_moments_iterative.py model=movielens method=vi'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _MVIIS_F --queue cnu --cmd 'python runner_moments_iterative.py model=movielens method=vi fake_data=True'

# lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _MRWSIS   --queue cnu --cmd 'python runner_moments_iterative.py model=movielens lrs=[0.1,0.03,0.01] method=rws'
# lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _MRWSIS_F --queue cnu --cmd 'python runner_moments_iterative.py model=movielens lrs=[0.1,0.03,0.01] method=rws fake_data=True'
