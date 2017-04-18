import subprocess
import sys
from os import listdir

if __name__=="__main__":
    fnames = listdir('legacy_tests/code/')
    for fname in fnames:
        if fname[-3:] == ".py":
            print(fname)
            path = 'legacy_tests/code/' + fname
            sys.stdout.flush()
            sys.stderr.flush()
            p = subprocess.Popen(['python', path], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stdout)
            sys.stdout.flush()
            sys.stderr.flush()
            p.wait()
