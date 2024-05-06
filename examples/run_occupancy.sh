lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n Occ_RWS --queue cnu --cmd 'python runner.py model=occupancy method=rws'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n Occ_QEM --queue cnu --cmd 'python runner.py model=occupancy method=qem'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n Occ_QEM_non_mp --queue cnu --cmd 'python runner.py model=occupancy method=qem non_mp=True'
