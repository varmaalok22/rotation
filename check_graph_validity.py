#Checking the validity of a graph as a transport system
import numpy as np
from numpy import linalg as LA
import timeit
import pylab

edge_range = []
node_range = []
edge_avg = []
num_edges_list = []
vertices = []

#Generating a graph (adjacency matrix) after seeding a pseudo random number generator
def Adj(size, threshold, seed):
    '''Produces an adjacency matrix of the prescribed size. To do so, it first makes a matrix full of
    random numbers between 0 and 1, using a set seed. Then, based on the threshold, it produces a
    binary matrix.'''
    np.random.seed(seed)
    adj_mat = np.random.rand(size,size)
    A = adj_mat > threshold
    A = A*1
    np.fill_diagonal(A,0)
    return A
    
#Checking Strong Connectivity of the Graph
def isStrong(adj_mat):
    edge_out = []
    edge_in = []
    for index, x in np.ndenumerate(adj_mat):
        if index[0] != index[1] and x == 1:
            if not index[0] in edge_out:
                edge_out.append(index[0])
            if not index[1] in edge_in:
                edge_in.append(index[1])
    if len(edge_out) == adj_mat.shape[0] and len(edge_in) == adj_mat.shape[0]:
        return True
    else:
        return False

#Function to find whether or not an adjacency matrix represents a connected graph or not.
#This is important if one is to find the "connectedness" of the graph.
def isConnected(adj_mat):
    #Find size ("shape") of the matrix.
    n = adj_mat.shape[0]
    #Now, add matrix to its transpose and the identity matrix, i.e. (A + A(T) + I).
    adj_tr = adj_mat.T
    identity = np.identity(n, dtype=int) #Generate an identity matrix with integer entries
    check = adj_mat + adj_tr + identity
    #Take the nth power of the matrix check
    check_n = LA.matrix_power(check, n)
    #See if check_n has any 0's in it. If it has a 0, the graph is disconnected
    test = check_n > 1
    test = test*1
    if 0 in test:
        return False
    else:
        return True
    
#To check if graph is 2-connected or not
def EdgeList(adj_mat):
    #To make a list of edges, take the indices of the non-zero entries from the adjacency matrix and store them as a tuple.
    edge_list = []
    for index, x in np.ndenumerate(adj_mat):
        if x != 0:
            edge_list.append(index)
    return edge_list
    
#Now to remove every possible pair of edges from the adjacency matrix and figure out its effect on connectivity.
def isTwoConnected(adj_mat):
    #For every edge in the list, remove every other edge and see if the graph remains connected.
    el = EdgeList(adj_mat)
    a = adj_mat[:] #Make a copy of the adjacency matrix
    test = []
    iden_edges = []
    for ind1, x in enumerate(el[:-1]): #Looping from first to second-last entry
        for ind2, y in enumerate(el[1+ind1:]): #Looping from ith to last entry
                a[x] = 0 #Remove edge1
                a[y] = 0 #Remove edge2
                test.append(isConnected(a))
                if isConnected(a) == False:
                    iden_edges.append((x,y))
                a[x] = 1 #Replace edge1
                a[y] = 1 #Replace edge2
    return (False in test, iden_edges)

#Now to combine all these helper functions into one function
#Given an adjacency matrix, it will return whether or not the input graph was valid
#If the graph is strongly connected, and two connected, it will set the identical edges to be zero
#(This is for the first computation only. Can be modified later.)
def isValidGraph(adj_mat):
    #First make a copy of adj_mat and only modify that!
    a = adj_mat[::]
    if isStrong(a):
        i = isTwoConnected(a)
        if i[0]:
            for e in i[1]: #For every identical edge in the matrix
                a[e[0]] = 0
                a[e[1]] = 0 #Remove the edges from the matrix
    return isStrong(a), a #See if the graph still remains strongly connected.

def AdjToInc(mat):
    '''This function converts an adjacency matrix into an incidence matrix'''
    n = np.shape(mat)[0] #This is the number of nodes in the graph
    k = 0 #This is the counter for the incidence matrix col
    m = 0 #This is the counter for the number of columns in the incidence matrix
    for index, x in np.ndenumerate(mat):
        m += x
    inc_mat = np.zeros(shape=(n,m),dtype=int)
    
    #Now we have to loop over the adjacency matrix to find the entries of the incidence matrix.
    for index, x in np.ndenumerate(mat):
        if x == 1:
            inc_mat[index[0],k] = -1
            inc_mat[index[1],k] = 1
            k += 1
    return inc_mat
    
#Converting an incidence matrix to an adjacency matrix
def IncToAdj(inc_mat):
    #Make a matrix where all the ones are (A)
    A = (inc_mat == 1)*1
    #Make another matrix containing the position of all -1's (B)
    B = (inc_mat == -1)*1
    #Dot product of A and B(tr)
    adj_mat = np.dot(B,A.T) #IMPORTANT TO TAKE CARE OF ORDER!!!
    return adj_mat
    
def CycleNumber(adj_mat):
    """Takes an adjacency matrix as the input, converts it into an incidence matrix.
    Then it applies Euler's formula to return the number of INDEPENDENT cycles (i.e. loops)
    in the graph."""
    inc = AdjToInc(adj_mat)
    m = np.shape(inc)[0]
    n = np.shape(inc)[1]
    #Euler's formula
    cyc_num = n - (m - 1)
    return cyc_num
    
'''To generate a plot of average score as a function of number of nodes and number of edges.'''
def GenGraphs(size, no_edges=None, sample_size=20):
    '''Generates a list with arrays of size nxn within it. These arrays
    represent valid strongly connected graphs with its 2-cuts removed, with a specific
    number of edges. 100 arrays need to be sampled.'''
    valid_graphs = []
    i = 0    
    n = 0
    while n<sample_size:
        a = Adj(size, 0.6, i)
        while isValidGraph(a)[0] == False:
            i += 1
            a = Adj(size, 0.6, i) #I am setting a threshold of 0.6 so that I have a higher chance of getting a valid graph.
	    #print i, a
        if no_edges != None:
            if np.sum(a) == no_edges:
                valid_graphs.append(a)
                n += 1
        else:
            valid_graphs.append(a)
            n += 1
        i += 1
	if i > 50000:
	    n = sample_size #This will safely break out of the loop if there is a chance of an infinite loop.
    return valid_graphs
    
'''Now that I have a valid list of graphs, I would like to calculate the 
number of valid paths in the graph for each graph in the list I have.
Then, I have to calculate the average number of paths.'''
def AdjMat2Dict(adj_mat):
    adj_mat_dict = {}
    for row_n in range(len(adj_mat)):
        connected_list = list(np.where(adj_mat[row_n])[0])
        adj_mat_dict.update({row_n: connected_list})
    return adj_mat_dict

def CountPaths(adj_mat_dict, source, target, visited_node_list=None):
    num_paths = 0
    if visited_node_list is None:
        visited_node_list = []
    if target in adj_mat_dict[source]:
        num_paths += 1
    visited_node_list.append(source)
    for new_source in adj_mat_dict[source]:
        if new_source not in visited_node_list and new_source != target:
            num_paths += CountPaths(adj_mat_dict, new_source, target, visited_node_list=visited_node_list[:])
    return num_paths

def NumPaths(adj_mat, source, target):
    adj_mat_dict = AdjMat2Dict(adj_mat)
    return CountPaths(adj_mat_dict, source, target)
    
#A = np.array([[0,1,1,0,1],[1,0,1,1,0],[0,1,0,1,0],[1,0,0,0,1],[0,0,1,1,0]])

def TotalPaths(adj_mat):
    n = np.shape(adj_mat)[0]
    count = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                count += NumPaths(adj_mat,i,j)
    return count
    
def AvgPaths(graph_list):
    scores = []
    for matrix in graph_list:
        scores.append(TotalPaths(matrix))
    avg = np.mean(scores)
    return round(avg,2)
    
'''A series of functions to help in getting my final output and in checking.'''    
def NumEdges(graph_list):
    no_edges = []
    for matrix in graph_list:
        no_edges.append(np.sum(matrix))
    return no_edges

def AvgEdges(lst):
    a = np.mean(lst)
    return round(a, 2)
    
def EdgesRange(lst):
    return min(lst), max(lst)

start_time = timeit.default_timer()
    
###'''For reference'''
###edge_range = []
###edge_avg = []
###num_edges_list = []
###for i in range(6,15,1):
###    graph_samples = GenGraphs(i)
###    num_edges = NumEdges(graph_samples)
###    num_edges_list.append(num_edges)
###    edge_range.append(EdgesRange(num_edges))
###    edge_avg.append(AvgEdges(num_edges))
###    
###    
###'''Complete dataset to plot.'''
###   
###elapsed = timeit.default_timer() - start_time
###
###'''vert = []
###for x in vertices:
###    vert.append(x[:])
###for ind, x in enumerate(vert):
###    x.append((edge_range[ind][1]+1, 0))
###    x.insert(0, (edge_range[ind][0]-1,0))
###    
###fig = plt.figure()
###ax = fig.add_subplot(111, projection='3d')
###
###def cc(arg):
###    return colorConverter.to_rgba(arg, alpha=0.6)
###    
###poly = PolyCollection(vert, facecolors=[cc('r'), cc('g'), cc('b')])
###                                         
###poly.set_alpha(0.7)
###ax.add_collection3d(poly, zs=xs, zdir='y')
###
###ax.set_xlabel('No. of edges')
###ax.set_xlim3d(0, 40)
###ax.set_ylabel('No. of nodes')
###ax.set_ylim3d(3, 8)
###ax.set_zlabel('Average no. of paths')
###ax.set_zlim3d(0,400)
###
###plt.show()
###
###
###def Scores(graph_list):
###    scores = []
###    for mat in graph_list:
###        scores.append(TotalPaths(mat))
###    return scores
###
###scores = Scores(GenGraphs(4, 8, 500))
###path_range = (min(scores),max(scores))
###no_paths = []
###no_graphs_per_path = []
###for i in range(path_range[0], path_range[1]+1):
###    no_paths.append(i)
###    no_graphs_per_path.append(scores.count(i))
###    
###plt.plot(no_paths,no_graphs_per_path, 'r--')'''
###

def generate_graphs( min_nodes, max_nodes ):
    global edge_range, node_range
    global num_edges_list, edge_avg
    global vertices
    node_range = range(min_nodes, max_nodes, 1)
    for i in node_range:
        print("[INFO] at i = %s" % i)
        graph_samples = GenGraphs(i)
        num_edges = NumEdges(graph_samples)
        num_edges_list.append(num_edges)
        edge_range.append(EdgesRange(num_edges))
        edge_avg.append(AvgEdges(num_edges))

    for ind, x in enumerate(node_range):
        no_paths = []
        ys = range(edge_range[ind][0],edge_range[ind][1]+1)
        for y in ys:
            print("[INFO] x=%s, y=%s" % (x, y))
            if y in num_edges_list[ind]:
                #Generate a sample set of graphs with i nodes and y edges
                sample_set = GenGraphs(x, y) 
                print("++ Sample set: %s" % sample_set[0] )
                no_paths.append(AvgPaths(sample_set))
        zs = list(zip(ys,no_paths))
        vertices.append(zs)
    with open('results.txt', 'w') as f:
        f.write('num_of_nodes : #paths\n')
        for i, v in enumerate(vertices):
            f.write('%s : %s\n' % (node_range[i], v))
    print("Done writing data to results.txt")

def main():
    generate_graphs( min_nodes = 5, max_nodes = 10 )

if __name__ == '__main__':
    main()
