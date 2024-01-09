from posteriordb import PosteriorDatabase
import os
pdb_path = os.path.join(os.getcwd(), "posteriordb/posterior_database")
my_pdb = PosteriorDatabase(pdb_path)

print(my_pdb.posterior_names())

posterior = my_pdb.posterior("radon_all-radon_pooled")

print(posterior.information)

print(posterior.model.code("stan"))

print(len(posterior.data.values()['county_idx']))