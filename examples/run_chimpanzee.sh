lbatch -c 1 -g 1 --gputype rtx_3090 -t 1 -m 64 -a !!!!!!!!!! -n MVI --queue cnu --cmd 'python runner.py model=chimpanzee method=vi'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 1 -m 64 -a !!!!!!!!!! -n MRWS --queue cnu --cmd 'python runner.py model=chimpanzee method=rws'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 1 -m 64 -a !!!!!!!!!! -n MQEM --queue cnu --cmd 'python runner.py model=chimpanzee method=qem'

lbatch -c 1 -g 1 --gputype rtx_3090 -t 1 -m 64 -a !!!!!!!!!! -n MQEM_non_mp --queue cnu --cmd 'python runner.py model=chimpanzee method=qem non_mp=True'