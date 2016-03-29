#!/usr/bin/env python

from __future__ import print_function
from past.builtins import execfile
import traceback
import re
from collections import Counter
import sys

# disable show in figures
import matplotlib.pyplot as plt
plt.show = lambda: None

statuscounter = Counter()

itemparser = re.compile('(?P<kind>.*) (?P<chapter>.*)\.(?P<number>.*)')

kinds = ['Figure', 'Example', 'Exercise']

FAILED = 'Failed'
NOTIMPLEMENTED = 'Not implemented'
SUCCESS = 'Success'

faillist = []

if __name__ == "__main__":
    with open('allitems.txt') as allitems:
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
            try:
                execfile(filename)
                status = SUCCESS
            except IOError:
                status = NOTIMPLEMENTED
            except Exception as err:
                status = FAILED
                message = traceback.format_exc()

            statuscounter[status] += 1

            if status != NOTIMPLEMENTED:
                print(kind, chapter, number, status)

            if status == FAILED:
                faillist.append([kind, chapter, number])
                print(message)

    for items in statuscounter.items():
        print("{}: {}".format(*items))
    print("Failed items:")
    for items in faillist:
        print("  {} {} {}".format(*items))

    sys.exit(statuscounter[FAILED])
