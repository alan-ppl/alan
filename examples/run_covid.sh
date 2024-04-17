cp covid/covid.py covid/results/covid.py

lbatch -c 1 -g 1 --gputype rtx_3090 -t 72 -m 64 -a cosc020762 -n CovVI --queue cnu --cmd 'python runner.py model=covid method=vi'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 72 -m 64 -a cosc020762 -n CovRWS --queue cnu --cmd 'python runner.py model=covid method=rws'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 120 -m 64 -a cosc020762 -n CovQEM --queue cnu --cmd 'python runner.py model=covid method=qem'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 120 -m 64 -a cosc020762 -n CovQEM_non_mp --queue cnu --cmd 'python runner.py model=covid method=qem non_mp=True'