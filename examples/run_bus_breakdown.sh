lbatch -c 1 -g 1 --gputype rtx_3090 -t 2 -m 64 -a !!!!!!!!!! -n BVI --queue cnu --cmd 'python runner.py model=bus_breakdown method=vi'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 2 -m 64 -a !!!!!!!!!! -n BRWS --queue cnu --cmd 'python runner.py model=bus_breakdown method=rws'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 2 -m 64 -a !!!!!!!!!! -n BQEM --queue cnu --cmd 'python runner.py model=bus_breakdown method=qem'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 2 -m 64 -a !!!!!!!!!! -n BQEM_non_mp --queue cnu --cmd 'python runner.py model=bus_breakdown method=qem non_mp=True'