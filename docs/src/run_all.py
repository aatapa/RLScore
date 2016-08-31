import subprocess
housing = ["housing_data", "regression1", "regression2", "regression3", "regression4", "regression5"]
parse = ["parse_data", "parse_regression1", "parse_regression2"]
adult = ["adult_data", "classification0", "classification1", "classification2", "classification3", "classification4"]

for fname in parse:
    f = open(fname+".out", 'w')
    x = subprocess.Popen(["python", fname+".py"], stdout=f)
    x.wait()
    f.close()

