lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _CVIIS   --queue cnu --cmd 'python runner_moments_iterative.py model=chimpanzees method=vi'
lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _CVIIS_F --queue cnu --cmd 'python runner_moments_iterative.py model=chimpanzees method=vi fake_data=True'

# lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _CRWSIS   --queue cnu --cmd 'python runner_moments_iterative.py model=chimpanzees method=rws'
# lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 5 -m 64 -a ********** -n _CRWSIS_F --queue cnu --cmd 'python runner_moments_iterative.py model=chimpanzees method=rws fake_data=True'
