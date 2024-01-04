echo "Running linear_gaussian.py"
python linear_gaussian.py
echo "Finished running linear_gaussian.py"

read -p "Press enter to continue."
echo ""

echo "Running linear_gaussian_plated.py"
python linear_gaussian_plated.py
echo "Finished running linear_gaussian_plated.py"

read -p "Press enter to continue."
echo ""

echo "Running predictive_example.py"
python predictive_example.py
echo "Finished running predictive_example.py"

read -p "Press enter to continue."
echo ""

echo "Running simple_elbo_experiment.py"
python simple_elbo_experiment.py
echo "Finished running simple_elbo_experiment.py"

read -p "Press enter to continue."
echo ""

echo "Running analytic_predictive_example.py"
cd analytic_pll
python analytic_predictive_example.py
echo "Finished running analytic_predictive_example.py"

read -p "Press enter to continue."
echo ""

echo "Running movielens.py"
cd ../movielens
python movielens.py
echo "Finished running movielens.py"

read -p "Press enter to continue."
echo ""

echo "Running bus_breakdown.py"
cd ../bus_breakdown
python bus_breakdown.py
echo "Finished running bus_breakdown.py"

echo ""
echo "Finished running all examples."
echo ""

# cd ..