# Note: this is using Ks_lrs = {30: [0.1, 0.03]}

lbatch -c 1 -g 1 --gputype rtx_3090 -t 20 -m 64 -a !!!!!!!!!! -n OccRep_RWS --queue cnu --cmd 'python runner.py model=occupancy_reparam method=rws reparam=False'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 20 -m 64 -a !!!!!!!!!! -n OccRep_QEM --queue cnu --cmd 'python runner.py model=occupancy_reparam method=qem reparam=False'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 20 -m 64 -a !!!!!!!!!! -n OccRep_QEM_non_mp --queue cnu --cmd 'python runner.py model=occupancy_reparam method=qem non_mp=True reparam=False'
