from __future__ import print_function
import numpy as np


def display_export_data(data, display_type, row_head, save=False, latex=False, width=None, sep='|'):
    """
    Create a table view of data. Data can also be exported for a csv file or
    LaTex tabular format. This function is ideal to display large amounts of
    data.

    Parameters
    ----------
    data : array or arrays
        The transfer function G(s) of the system. The first item (data[0]) in
        the array must be an array of independent variables. From the second
        item onward (data[1], data[2], ...), an array of dependent variables
        are defined. The dependent varaiable array should be defined as an
        array itself.
    display_type : string
        Choose the main display label and file name to use.
    row_head : array
        A list is row headings for the depended variables.
    save : boolean
        If true, a csv data file is saved, with wthe name pz_type.csv
        (optional).
    latex : boolean
        If true, the data file is further converted to LaTex tabular format
        (optional).
    width : integer
        If the width of the rows exceed the width of a page, this number will
        limits the number of items to be displayed in a row. Multple tables are
        created. (optional).
    sep : char
        Specify the separator operator to use in the table (optional).

    Returns
    -------
    File : csv file
    Print : screen output
    """

    if latex:
        save = True  # a file needs to be saved for Latex output
        f = open(display_type + '.tex', 'wb')
    elif save: f = open(display_type + '.csv', 'wb')

    row_heading = []  # array to store heading labels
    n = np.shape(row_head)[0]  # number of types of headings

    for i in range(n):
        m = len(data[0][i + 1])  # number of items for heading type
        for j in range(m):
            if m > 1:  # increment heading label
                if latex:
                    row_heading.append('${0}_{1}$'.format(row_head[i], j + 1))
                else:
                    row_heading.append('{0}{1}'.format(row_head[i], j + 1))
            else:
                if latex:
                    row_heading.append('${0}$'.format(row_head[i]))
                else:
                    row_heading.append('{0}'.format(row_head[i]))

    if width is not None:  # separate the data in sections for the specified width
        sec = []
        m = len(data)
        section = m / width
        sectionlast = m % width
        if sectionlast != 0:  # the last section of data might be less than the specified width
            section += 1

        for s in range(section):
            sec.append(data[width * s:width * (s + 1)])
    else:
        sec = [data]

    for s in sec:  # cycle through all the data sections
        top = display_type  # main label
        tabs = ''  # used for LaTex format only

        rows = row_heading[:]  # reinitialise the main data array with heading labels

        o = len(s)  # number if columns in the section
        for i in range(o):  # cycle through columns
            top += ' ' + sep + ' {:.3e}'.format(s[i][0])  # format independent variables
            row_count = 0
            for j in range(n):  # cycle through row headings
                m = len(data[0][j + 1])  # each heading type count could be different
                for k in range(m):  # cycle through items in heading type
                    u = s[i][j + 1][k]  # extract data
                    if isinstance(u, (float)):  # data is float
                        u = '{:.3e}'.format(u)
                    elif isinstance(u, (str, bool, int)): # data is string or boolean
                        u = ' {}'.format(u)
                    else:  # data is matrix
                        if latex:  # improves formatting
                            if u.imag == 0:
                                u = ' \\num{' + '{:.3e}'.format(u[0, 0]) + '}'
                            else:
                                u = ' \\num{' + '{:.3e}'.format(u[0, 0].real) + '}\
                                 \\num{' + '{:.3e}'.format(u[0, 0].imag) + '}i'
                        else:
                            if u.imag == 0:
                                u = '{:.3e}'.format(u[0, 0])
                            else:
                                u = '{:.3e}'.format(u[0, 0].real) + '{:.3e}'.format(u[0, 0].imag)
                    rows[row_count] += ' ' + sep + u  # format dependent variable
                    row_count += 1

            tabs += 'c '

        if latex:
            header = '\\begin{tabular}{%sc}\n' % tabs  # add an extra c
            header += '\\toprule\n'
            header += top + '\\\\\n'
            header += '\\midrule\n'
            f.write(header)
        elif save: f.write(top + '\n')
        if not latex: print(top)
        if not latex: print('')
        for i in range(len(rows)):
            if not latex: print(rows[i])
            if save: f.write(rows[i] + '\\\\\n')
        if not latex: print('')
        if latex:
            footer = '\\bottomrule\n'
            footer += '\\end{tabular}\n\n'
            f.write(footer)

    if save: f.close()
