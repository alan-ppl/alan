lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n MVI_reparam --queue cnu --cmd 'python runner.py model=movielens_reparam method=vi split.plate=plate_1 split.size=20'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n MRWS_reparam --queue cnu --cmd 'python runner.py model=movielens_reparam method=rws split.plate=plate_1 split.size=20'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n MQEM_reparam --queue cnu --cmd 'python runner.py model=movielens_reparam method=qem split.plate=plate_1 split.size=20'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n MQEM_non_mp_reparam --queue cnu --cmd 'python runner.py model=movielens_reparam method=qem non_mp=True split.plate=plate_1 split.size=20'