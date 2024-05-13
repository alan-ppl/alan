# Note: this is using Ks_lrs = {30: [0.1, 0.03]}

lbatch -c 1 -g 1 --gputype rtx_3090 -t 7 -m 64 -a !!!!!!!!!! -n RadonRepVI --queue cnu --cmd 'python runner.py model=radon_reparam method=vi save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 7 -m 64 -a !!!!!!!!!! -n RadonRepRWS --queue cnu --cmd 'python runner.py model=radon_reparam method=rws save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 7 -m 64 -a !!!!!!!!!! -n RadonRepQEM --queue cnu --cmd 'python runner.py model=radon_reparam method=qem save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 7 -m 64 -a !!!!!!!!!! -n RadonRepQEM_non_mp --queue cnu --cmd 'python runner.py model=radon_reparam method=qem non_mp=True save_moments=True'
