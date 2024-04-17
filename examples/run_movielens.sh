lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 1 -m 64 -a !!!!!!!!!! -n MVI --queue cnu --cmd 'python runner.py model=movielens method=vi'

lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 1 -m 64 -a !!!!!!!!!! -n MRWS --queue cnu --cmd 'python runner.py model=movielens method=rws'

lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 1 -m 64 -a !!!!!!!!!! -n MQEM --queue cnu --cmd 'python runner.py model=movielens method=qem'

lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 1 -m 64 -a !!!!!!!!!! -n MQEM_non_mp --queue cnu --cmd 'python runner.py model=movielens method=qem non_mp=True'