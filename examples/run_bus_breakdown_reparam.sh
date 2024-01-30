lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 2 -m 64 -a !!!!!!!!!! -n BVI --queue cnu --cmd 'python runner.py model=bus_breakdown_reparam method=vi'

lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 2 -m 64 -a !!!!!!!!!! -n BRWS --queue cnu --cmd 'python runner.py model=bus_breakdown_reparam method=rws lrs=[0.0003,0.0001,0.00003]'

lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 2 -m 64 -a !!!!!!!!!! -n BQEM --queue cnu --cmd 'python runner.py model=bus_breakdown_reparam method=qem lrs=[0.0003,0.0001,0.00003]'