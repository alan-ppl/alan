For eight schools and radon:

```bash
git clone https://github.com/stan-dev/posteriordb
pip install posteriordb
```

For the UCI datasets:
    
```bash
mkdir data
wget https://archive.ics.uci.edu/static/public/143/statlog+australian+credit+approval.zip
unzip statlog+australian+credit+approval.zip -d data

wget https://archive.ics.uci.edu/static/public/151/connectionist+bench+sonar+mines+vs+rocks.zip
unzip connectionist+bench+sonar+mines+vs+rocks.zip -d data

cd data/
wget "https://esapubs.org/archive/ecol/E087/050/butterflyData.txt"
```