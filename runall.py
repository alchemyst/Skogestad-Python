#!/usr/bin/env python

import glob
import os
import traceback
from operator import itemgetter
import re

# disable show in figures
import matplotlib.pyplot as plt
plt.show = lambda: None

itemparser = re.compile('(?P<kind>.*) (?P<chapter>.*)\.(?P<number>.*)')
allitems = open('allitems.txt').read().splitlines()

kinds = ['Figure', 'Example', 'Exercise']

statustable = []


def runfile(f):
    try:
        execfile(f)
        return 'Success', None
    except Exception, err:
        return 'Failed', traceback.format_exc()


if __name__ == "__main__":
    for item in allitems:
        kind, chapter_c, number_c = itemparser.match(item).groups()

        status = 'Not implemented'

        number = int(number_c)

        if chapter_c.isdigit():
            chapter = int(chapter_c)
            mask = '{}_{:02d}_{:02d}.py'
        else:
            chapter = chapter_c
            mask = '{}_{}_{}.py'

        filename = mask.format(kind, chapter, number)
        if os.path.exists(filename):
            status, message = runfile(filename)

        print kind, chapter, number, status
        if status == 'Failed':
            print message
