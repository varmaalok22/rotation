import sys
import matplotlib.pyplot as plt
import numpy as np

def main( ):
    res = sys.argv[1]
    with open(res, 'r') as f:
        txt = f.read()
    lines = txt.split("\n")
    zs = []
    verts = []
    for l in lines:
        if not l.strip():
            continue
        numNode, paths = l.split(':')
        data = eval( paths )
        xvec, yvec = zip(*data)
        plt.semilogy(xvec, yvec, label='#nodes=%s' % numNode)
    plt.legend(loc='best', framealpha=0.4)
    plt.show()

if __name__ == '__main__':
    main()
