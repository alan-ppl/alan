lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 1 -m 64 -a !!!!!!!!!! -n MVI --queue cnu --cmd 'python runner.py model=radon method=vi'

lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 1 -m 64 -a !!!!!!!!!! -n MRWS --queue cnu --cmd 'python runner.py model=radon method=rws'

lbatch -c 1 -g 1 --gputype A100 --exclude_40G_A100 -t 1 -m 64 -a !!!!!!!!!! -n MQEM --queue cnu --cmd 'python runner.py model=radon method=qem'
