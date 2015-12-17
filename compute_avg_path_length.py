# This script computes average path length.

import sys
import os
import numpy as np
import math
from methods import *

avg_min_ = []
result_file_ = 'results.csv'
with open( result_file_, 'w') as f:
    f.write('data_file, dimens, avg_min\n')

def process_data_file( dataFile ):
    print( "[INFO] Processing %s" % dataFile )
    global avg_min_
    with open( dataFile, "r") as f:
        txt = f.read()
    mats = filter(None, txt.split('\n\n\n'))
    graph_list = []
    for m in mats:
        m = np.matrix( m )
        dimen = int(math.sqrt(m.shape[1]))
        mat = np.ndarray(shape = (dimen, dimen), dtype=int, buffer = m )
        graph_list.append( mat )
    avgMin = AvgMin( graph_list )
    avg_min_.append( avgMin )
    with open(result_file_, 'a') as f:
        f.write("%s, %s, %s\n" % (dataFile, dimen, avgMin ))

def compute( dataDir ):
    print("[INFO] Processing data in %s" % dataDir )
    dataFiles = []
    for d, sd, fs in os.walk( dataDir ):
        for f in fs:
            if '.dat' in f:
                dataFiles.append( os.path.join(d, f) )
    [ process_data_file( f ) for f in sorted(dataFiles) ]


def main( ):
    dataDir = sys.argv[1]
    compute( dataDir )
    

if __name__ == "__main__":
    main( )
