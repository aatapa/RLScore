import subprocess
import sys
from os import listdir

if __name__=="__main__":
    fnames = listdir('examples/code/')
    for fname in fnames:
        if fname[-3:] == ".py":
            print fname
            path = 'examples/code/' + fname
            sys.stdout.flush()
            sys.stderr.flush()
            p = subprocess.Popen(['python', path], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stdout)
            sys.stdout.flush()
            sys.stderr.flush()
            p.wait()
