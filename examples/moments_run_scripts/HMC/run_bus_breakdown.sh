lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 6 -m 64 -a ********** -n _BHMC   --queue cnu --cmd 'python runner_moments_HMC.py model=bus_breakdown'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 6 -m 64 -a ********** -n _BHMC_F --queue cnu --cmd 'python runner_moments_HMC.py model=bus_breakdown fake_data=True'
