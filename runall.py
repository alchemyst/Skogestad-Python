#!/usr/bin/env python

from __future__ import print_function
from past.builtins import execfile
import traceback
import re
from collections import Counter
import sys
import time
import logging

logging.basicConfig(level=logging.DEBUG)

# disable show in figures
import matplotlib.pyplot as plt
plt.show = lambda: None

# disable print
def print(*args):
    pass

statuscounter = Counter()
times = Counter()

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
            starttime = time.time()
            try:
                execfile(filename)
                status = SUCCESS
            except IOError:
                status = NOTIMPLEMENTED
            except Exception as err:
                status = FAILED
                message = traceback.format_exc()

            times[(kind, chapter, number)] = time.time() - starttime

            statuscounter[status] += 1

            if status != NOTIMPLEMENTED:
                logging.info(' '.join(str(thing) for thing in [kind, chapter, number, status]))

            if status == FAILED:
                faillist.append([kind, chapter, number])
                logging.warning(message)

    for items in statuscounter.items():
        logging.info("{}: {}".format(*items))
    logging.info("Failed items:")
    for items in faillist:
        logging.info("  {} {} {}".format(*items))

    logging.info("Slowest tests")
    for [(kind, chapter, number), elapsed] in times.most_common(5):
        logging.info("    {} {} {} {}".format(kind, chapter, number, elapsed))
    sys.exit(statuscounter[FAILED])
