#!/usr/bin/env python

import os
import traceback
import re
from collections import Counter
import sys

# disable show in figures
import matplotlib.pyplot as plt
plt.show = lambda: None

statuscounter = Counter()

itemparser = re.compile('(?P<kind>.*) (?P<chapter>.*)\.(?P<number>.*)')
allitems = open('allitems.txt').read().splitlines()

kinds = ['Figure', 'Example', 'Exercise']

faillist = []

if __name__ == "__main__":
    for item in allitems:
        kind, chapter_c, number_c = itemparser.match(item).groups()

        number = int(number_c)

        if chapter_c.isdigit():
            chapter = int(chapter_c)
            mask = '{}_{:02d}_{:02d}.py'
        else:
            chapter = chapter_c
            mask = '{}_{}_{}.py'

        filename = mask.format(kind, chapter, number)
        if os.path.exists(filename):
            try:
                execfile(filename)
                status = 'Success'
            except Exception, err:
                status = 'Failed'
                message = traceback.format_exc()
        else:
            status = 'Not implemented'

        statuscounter[status] += 1

        if status != 'Not implemented':
            print kind, chapter, number, status

        if status == 'Failed':
            faillist.append([kind, chapter, number])
            print message

    for items in statuscounter.iteritems():
        print "{}: {}".format(*items)
    print "Failed items:"
    for items in faillist:
        print "  {} {} {}".format(*items)

    sys.exit(statuscounter['Failed'])
