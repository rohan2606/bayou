import os
import shutil
import sys

def rm_r(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

def cleanUp(logdir = "./log/"):

    print("Cleaning all files in log ... ", end="")
    sys.stdout.flush()

    rm_r(logdir)

    # for f in os.listdir(logdir):
    #     rm_r(os.path.join(logdir, f))

    os.mkdir(logdir)
    os.mkdir(logdir + "/JSONFiles")
    print("Done")
