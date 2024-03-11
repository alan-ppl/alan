lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _CVIIS10   --queue cnu --cmd 'python runner_moments_iterative.py model=chimpanzees K=10 method=vi'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _CVIIS10_F --queue cnu --cmd 'python runner_moments_iterative.py model=chimpanzees K=10 method=vi fake_data=True'

lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _CRWSIS10   --queue cnu --cmd 'python runner_moments_iterative.py model=chimpanzees K=10 method=rws'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _CRWSIS10_F --queue cnu --cmd 'python runner_moments_iterative.py model=chimpanzees K=10 method=rws fake_data=True'
