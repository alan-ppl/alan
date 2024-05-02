lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n BVI --queue cnu --cmd 'python runner.py model=bus_breakdown method=vi save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n BRWS --queue cnu --cmd 'python runner.py model=bus_breakdown method=rws save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n BQEM --queue cnu --cmd 'python runner.py model=bus_breakdown method=qem save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n BQEM_non_mp --queue cnu --cmd 'python runner.py model=bus_breakdown method=qem non_mp=True save_moments=True'