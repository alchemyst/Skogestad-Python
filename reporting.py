import numpy as np

from Chapter_04 import pole_zero_directions


def export_pole_zero_data(G, vec, pz_type, save=False, latex=False, width=None, separator='&'):
    '''
    Create a table view of direction data (input and output) for either poles
    and zeros. Data can also be exported for a csv file or formated to Latex
    tabular format. This function is ideal to display large amounts of data.
    
    Parameters
    ----------
    G : numpy matrix
        The transfer function G(s) of the system.
    vec : numpy array
        A vector containing all the transmission poles or zeros of a system.
    pz_type : string ['Pole','Zero']
        Choose the display label and file name to use.
    save : boolean
        If true, a csv data file is saved, with wthe name pz_type.csv (optional).
    latex : boolean
        If true, the data file is further converted to Latex tabular format
        (optional).
    width : integer
        If the width of the rows exceed the width of a page, this number will
        limits the number of items to be displayed in a row. Multple tables are
        created. (optional).
    Separator : char
        Specify the separator operator to use in the table (optional).
    
    Returns
    -------
    File : csv file
    Print : screen output
    '''
    
    if latex: save = True
    if save: f = open(pz_type + ".csv", "wb")
    
    inputlab = []
    outputlab = [] 
    n = np.shape(G(1))[0]
    
    for j in range(0, n):
        if latex:
            inputlab.append('$u_%s$' % j)
            outputlab.append('$y_%s$' % j)
        else:
            inputlab.append('u%s' % j)
            outputlab.append('y%s' % j)
    
    if not width is None:
        pz_sec = []
        m = np.shape(vec)[0]
        section = m / width
        sectionlast = m % width
        if sectionlast != 0:
            section += 1
        
        for s in range(0, section):
            pz_sec.append(vec[width * s:width * (s + 1)])
    else:
        pz_sec = [vec]
    
    for pz in pz_sec:
        vec_sec = pole_zero_directions(G, pz)
        
        top = pz_type
        tabs = ''    
        
        inputs = outputlab[:] 
        outputs = inputlab[:]
        
        m = np.shape(pz)[0]
        for i in range(0, m):
            top +=  ' ' + separator + ' {:.3e}'.format(pz[i])
            for j in range(0, n):
                u = vec_sec[i][1][j]
                inputs[j] += ' ' + separator + ' {:.3f}'.format(u[0,0])
                
                y = vec_sec[i][2][j]
                outputs[j] += ' ' + separator + ' {:.3f}'.format(y[0,0])
    
            tabs += 'c ' #latex only      
        
        if latex:
            header = '\\begin{tabular}{%sc}\n' % tabs # add an extra c
            header += '\\toprule\n'
            header += top + '\\\\\n'
            header += '\\midrule'
            f.write(header)
        elif save: f.write(top + '\n')
        print top
        print ''
        for j in range(0, n):
            print inputs[j]
            print ''
            if save: f.write(inputs[j] + '\\\\\n')
        for j in range(0, n):
            print outputs[j]
            print ''
            if save: f.write(outputs[j] + '\\\\\n')
        if latex:
            footer = '\\bottomrule\n'
            footer += '\\end{tabular}\n\n'
            f.write(footer)
    
    if save: f.close()
