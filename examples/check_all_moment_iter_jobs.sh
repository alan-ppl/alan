echo "bus vi REAL"
python runner_moments_iterative.py model=bus_breakdown method=vi lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=False write_job_status=False
read -p "Press enter to continue."
echo ""

echo "bus vi FAKE"
python runner_moments_iterative.py model=bus_breakdown method=vi lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=True write_job_status=False
read -p "Press enter to continue."
echo ""

echo "bus rws REAL"
python runner_moments_iterative.py model=bus_breakdown method=rws lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=False write_job_status=False
read -p "Press enter to continue."
echo ""

echo "bus rws FAKE"
python runner_moments_iterative.py model=bus_breakdown method=rws lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=True write_job_status=False
read -p "Press enter to continue."
echo ""



echo "chimpanzees vi REAL"
python runner_moments_iterative.py model=chimpanzees method=vi lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=False write_job_status=False
read -p "Press enter to continue."
echo ""

echo "chimpanzees vi FAKE"
python runner_moments_iterative.py model=chimpanzees method=vi lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=True write_job_status=False
read -p "Press enter to continue."
echo ""

echo "chimpanzees rws REAL"
python runner_moments_iterative.py model=chimpanzees method=rws lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=False write_job_status=False
read -p "Press enter to continue."
echo ""

echo "chimpanzees rws FAKE"
python runner_moments_iterative.py model=chimpanzees method=rws lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=True write_job_status=False
read -p "Press enter to continue."
echo ""




echo "movielens vi REAL"
python runner_moments_iterative.py model=movielens method=vi lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=False write_job_status=False
read -p "Press enter to continue."
echo ""

echo "movielens vi FAKE"
python runner_moments_iterative.py model=movielens method=vi lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=True write_job_status=False
read -p "Press enter to continue."
echo ""

echo "movielens rws REAL"
python runner_moments_iterative.py model=movielens method=rws lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=False write_job_status=False
read -p "Press enter to continue."
echo ""

echo "movielens rws FAKE"
python runner_moments_iterative.py model=movielens method=rws lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=True fake_data=True write_job_status=False
read -p "Press enter to continue."
echo ""


echo "occupancy rws REAL"
python runner_moments_iterative.py model=occupancy method=rws lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=False fake_data=False write_job_status=False
read -p "Press enter to continue."
echo ""

echo "occupancy rws FAKE"
python runner_moments_iterative.py model=occupancy method=rws lrs=[0.3,0.1] num_runs=3 num_iters=10 reparam=False fake_data=True write_job_status=False
read -p "Press enter to continue."
echo ""