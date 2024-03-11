lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _BVIIS   --queue cnu --cmd 'python runner_moments_iterative.py model=bus_breakdown method=vi'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _BVIIS_F --queue cnu --cmd 'python runner_moments_iterative.py model=bus_breakdown method=vi fake_data=True'

# lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _BRWSIS   --queue cnu --cmd 'python runner_moments_iterative.py model=bus_breakdown lrs=[0.1,0.03,0.01] method=rws'
# lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _BRWSIS_F --queue cnu --cmd 'python runner_moments_iterative.py model=bus_breakdown lrs=[0.1,0.03,0.01] method=rws fake_data=True'
