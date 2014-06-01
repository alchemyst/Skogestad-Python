#!/usr/bin/env python

import sys
import glob
from matplotlib import pyplot as plt

# disable show in figures
plt.show = lambda: False

types = ['Figure', 'Example', 'Exercise']

if __name__ == "__main__":
    for t in types:
        files = glob.glob(t + "*.py")
        print "Running", len(files), t + "s"
        for f in files:
            print " ", f
            execfile(f)
