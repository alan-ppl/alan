lbatch -c 1 -g 1 --gputype rtx_3090 -t 72 -m 64 -a !!!!!!!!!! -n CovVI --queue cnu --cmd 'python runner.py model=covid method=vi save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 72 -m 64 -a !!!!!!!!!! -n CovRWS --queue cnu --cmd 'python runner.py model=covid method=rws save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 120 -m 64 -a !!!!!!!!!! -n CovQEM --queue cnu --cmd 'python runner.py model=covid method=qem save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 120 -m 64 -a !!!!!!!!!! -n CovQEM_non_mp --queue cnu --cmd 'python runner.py model=covid method=qem non_mp=True save_moments=True'