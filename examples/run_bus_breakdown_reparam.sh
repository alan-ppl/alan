lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n BVI_reparam --queue cnu --cmd 'python runner.py model=bus_breakdown_reparam method=vi save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n BRWS_reparam --queue cnu --cmd 'python runner.py model=bus_breakdown_reparam method=rws save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n BQEM_reparam --queue cnu --cmd 'python runner.py model=bus_breakdown_reparam method=qem save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n BQEM_non_mp_reparam --queue cnu --cmd 'python runner.py model=bus_breakdown_reparam method=qem non_mp=True save_moments=True'