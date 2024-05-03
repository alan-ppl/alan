lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n RadonVI --queue cnu --cmd 'python runner.py model=radon method=vi save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n RadonRWS --queue cnu --cmd 'python runner.py model=radon method=rws save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n RadonQEM --queue cnu --cmd 'python runner.py model=radon method=qem save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n RadonQEM_non_mp --queue cnu --cmd 'python runner.py model=radon method=qem non_mp=True save_moments=True'
