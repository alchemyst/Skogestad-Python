#!/usr/bin/env python

import sys
import glob
import traceback
from operator import itemgetter

# disable show in figures
from matplotlib import pyplot as plt
plt.show = lambda: None


types = ['Figure', 'Example', 'Exercise']

statustable = []

if __name__ == "__main__":
    for t in types:
        files = glob.glob(t + "*.py")
        print "Running", len(files), t + "s"
        for f in files:
            print " ", f
            try:
                execfile(f)
                statustable.append([f, 'Success'])
            except Exception, err:
                print traceback.format_exc()
                statustable.append([f, 'Failed'])

    statustable.sort(key=itemgetter(1))
    for filename, status in statustable:
        print filename, status