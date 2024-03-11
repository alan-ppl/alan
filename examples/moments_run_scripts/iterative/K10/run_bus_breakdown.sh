lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _BVIIS10   --queue cnu --cmd 'python runner_moments_iterative.py model=bus_breakdown K=10 method=vi'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _BVIIS10_F --queue cnu --cmd 'python runner_moments_iterative.py model=bus_breakdown K=10 method=vi fake_data=True'

lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _BRWSIS10   --queue cnu --cmd 'python runner_moments_iterative.py model=bus_breakdown K=10 lrs=[0.3,0.1,0.03,0.01] method=rws'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 10 -m 64 -a ********** -n _BRWSIS10_F --queue cnu --cmd 'python runner_moments_iterative.py model=bus_breakdown K=10 lrs=[0.3,0.1,0.03,0.01] method=rws fake_data=True'
