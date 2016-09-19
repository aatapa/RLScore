import os
import subprocess

fnames = os.listdir(".")
fnames.sort()
for fname in fnames:
    if fname[-2:] == "py" and fname not in ["run_all.py", "parse_regression_plot.py"]:
        print fname
        f = open(fname[:-3]+".out", 'w')
        x = subprocess.Popen(["python", fname], stdout=f)
        x.wait()
        f.close()

