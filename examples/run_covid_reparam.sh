lbatch -c 1 -g 1 --gputype rtx_3090 -t 72 -m 64 -a !!!!!!!!!! -n CovVI --queue cnu --cmd 'python runner.py model=covid_reparam method=vi save_moments=True num_iters=2000'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 72 -m 64 -a !!!!!!!!!! -n CovRWS --queue cnu --cmd 'python runner.py model=covid_reparam method=rws save_moments=True num_iters=2000'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 120 -m 64 -a !!!!!!!!!! -n CovQEM --queue cnu --cmd 'python runner.py model=covid_reparam method=qem save_moments=True num_iters=2000'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 120 -m 64 -a !!!!!!!!!! -n CovQEM_non_mp --queue cnu --cmd 'python runner.py model=covid_reparam method=qem non_mp=True save_moments=True num_iters=2000'