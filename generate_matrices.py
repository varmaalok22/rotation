"""
This script generate data
"""
from methods import *
mat_dir = '_matrices'
if os.path.isdir( os.path.join( '.', mat_dir)):
    shutil.rmtree( mat_dir )
os.makedirs( mat_dir )

def main():
    start_time = timeit.default_timer()
    all_graphs = []
    for i in range(5,14,1):
        graph_samples = GenGraphs(i, sample_size=20)
        all_graphs.append(graph_samples)
        outfile = os.path.join( mat_dir, 'mat_%s.dat' % i )
        for mat in graph_samples:
            with open( outfile, "a") as f:
                print("[INFO] Writing matrix %s to %s" % (i, outfile))
                f.write( np.array2string( mat , separator=','))
                f.write("\n\n\n")

    avg_min = []
    for graph_list in all_graphs[5:]:
        avg_min.append(AvgMin(graph_list))
    
    elapsed = timeit.default_timer() - start_time


if __name__ == "__main__":
    main()



