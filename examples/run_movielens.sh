lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n MVI --queue cnu --cmd 'python runner.py model=movielens method=vi split.plate=plate_1 split.size=20 save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n MRWS --queue cnu --cmd 'python runner.py model=movielens method=rws split.plate=plate_1 split.size=20 save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n MQEM --queue cnu --cmd 'python runner.py model=movielens method=qem split.plate=plate_1 split.size=20 save_moments=True'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 60 -m 64 -a !!!!!!!!!! -n MQEM_non_mp --queue cnu --cmd 'python runner.py model=movielens method=qem non_mp=True split.plate=plate_1 split.size=20 save_moments=True'