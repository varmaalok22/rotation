import numpy as np
from numpy import linalg as LA
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import timeit
#import itertools as it

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

'''A much, much better implementation of strong connectivity.'''
def strongly_connected_components(graph):
    """
    Tarjan's Algorithm (named for its discoverer, Robert Tarjan) is a graph theory algorithm
    for finding the strongly connected components of a graph.
    
    Based on: http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    
    Source: Dries Verdegem (http://www.logarithmic.net/pfh/blog/01208083168)    
    """
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    result = []
    
    def strongconnect(node):
        # set the depth index for this node to the smallest unused index
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        # Consider successors of `node`
        try:
            successors = graph[node]
        except:
            successors = []
        for successor in successors:
            if successor not in lowlinks:
                # Successor has not yet been visited; recurse on it.
                strongconnect(successor)
                lowlinks[node] = min(lowlinks[node],lowlinks[successor])
            elif successor in stack:
                # the successor is in the stack and hence in the current strongly connected component (SCC)
                lowlinks[node] = min(lowlinks[node],index[successor])
        # If `node` is a root node, pop the stack and generate an SCC
        if lowlinks[node] == index[node]:
            connected_component = []       
            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node: break
            component = tuple(connected_component)
            # storing the result
            result.append(component)
    for node in graph:
        if node not in lowlinks:
            strongconnect(node)
    return result

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
                mat_dict = AdjMat2Dict(a)
                t = (len(strongly_connected_components(mat_dict)) == 1)
                test.append(t)
                if t == False:
                    iden_edges.append((x,y))
                a[x] = 1 #Replace edge1
                a[y] = 1 #Replace edge2
    return (False in test, iden_edges)

def isCompletelyConnected(adj_mat):
    test = []
    for ind, x in np.ndenumerate(adj_mat):
        if ind[0] != ind[1] and x == 0:
            test.append(False)
    return False not in test

#Now to combine all these helper functions into one function
#Given an adjacency matrix, it will return whether or not the input graph was valid
#If the graph is strongly connected, and two connected, it will set the identical edges to be zero
#(This is for the first computation only. Can be modified later.)
def isValidGraph(adj_mat):
    '''This function returns 3 things - whether or not the graph is valid,
    the matrix after all the modifications, and a list of identical edges that
    were removed.'''
    #First make a copy of adj_mat and only modify that!
    import copy
    a = copy.deepcopy(adj_mat) #Very important, since this makes a copy of the matrix and doesn't change the original
    adj_mat_dict = AdjMat2Dict(a)    
    if isCompletelyConnected(a):
        return False, a, []
    if len(strongly_connected_components(adj_mat_dict))==1:
        i = isTwoConnected(a)
        if i[0]:
            for e in i[1]: #For every identical edge in the matrix
                if e[0][0] == e[1][0] or e[0][1] == e[1][1]:
                    return False, a, [] #This is a case when both the 2-cut edges are pointing in the same direction!!
                a[e[0]] = 0
                a[e[1]] = 0 #Remove the identical 2-cut-causing edges from the matrix
                if e[0][0] != e[1][1] and e[1][0] != e[0][1]: #NO SELF_LOOPS!
                    a[e[0][0], e[1][1]] = 1 #Correct the edges...
                    a[e[1][0], e[0][1]] = 1
        #if len(strongly_connected_components(AdjMat2Dict(a))) == 2:
        return True, a, i[1]
    else:
        return False, a, []
        
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
    
'''To generate a plot of average score as a function of number of nodes and number of edges.'''
def GenGraphs(size, no_edges=None, sample_size=100, threshold=0.4):
    '''Generates a list with arrays of size nxn within it. These arrays
    represent valid strongly connected graphs with its 2-cuts removed, with a specific
    number of edges. 100 arrays need to be sampled.'''
    valid_graphs = []
    #iden_edges = []
    seed = 0
    number_samples = 0
    a = Adj(size, threshold, seed)
    #val = isValidGraph(a)
    while number_samples < sample_size:
        print("[INFO] %s" % number_samples)
        a = Adj(size, threshold, seed)
        if no_edges is None:
            while not isValidGraph(a)[0]:
                seed += 1
                a = Adj(size, threshold, seed)
                #print seed, a            
                #val = isValidGraph(a)
            #valid_graphs.append((isValidGraph(a)[1], isValidGraph(a)[2]))
            #print valid_graphs
            valid_graphs.append(isValidGraph(a)[1]) #For now just look at the matrices that come out of this.
            seed += 1        
            number_samples += 1
        else:
            while not isValidGraph(a)[0] and np.sum(isValidGraph(a)[1]) != no_edges:
                seed += 1
                a = Adj(size, threshold, seed)
                #print seed, a            
                #val = isValidGraph(a)
                #valid_graphs.append((isValidGraph(a)[1], isValidGraph(a)[2]))
                #print valid_graphs
            valid_graphs.append(isValidGraph(a)[1]) #For now just look at the matrices that come out of this.
            seed += 1        
            number_samples += 1
        if size == 5 and len(valid_graphs)==100:
            valid_graphs.__delitem__(99)
    return valid_graphs
    
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

def TotalPaths(adj_mat):
    n = np.shape(adj_mat)[0]
    count = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                count += NumPaths(adj_mat,i,j)
    return count
    
def ListPaths(adj_mat_dict, source, target, visited_node_list=None, path_list=None):
    if path_list is None:
        path_list = []
    if visited_node_list is None:
        visited_node_list = []
    '''if path_list == []:
        pass
    elif target in adj_mat_dict[source]:
        path_list.append(target)'''
    visited_node_list.append(source)
    for new_source in adj_mat_dict[source]:
        if new_source not in visited_node_list and new_source != target:
            path_list=ListPaths(adj_mat_dict, new_source, target, visited_node_list=visited_node_list[:],path_list=path_list[:])
        elif new_source == target:
            visited_node_list.append(target)
    path_list.append(visited_node_list)
    for x in path_list:
        if x[-1] != target:
            path_list.remove(x)
    return path_list
    
def MinPathLen(adj_mat_dict, source, target):
    if target in adj_mat_dict[source]:
        return 1
    else:
        path_list = ListPaths(adj_mat_dict, source, target)
        path_lens = []        
        for x in path_list:
            path_lens.append(len(x) - 1) #The minus one is because the list has both the start point and the end point, which is redundant.
        return min(path_lens)

def NodePairs(adj_mat_dict):
    '''Takes an adjacency matrix dictionary and then outputs a list of tuples of
    ordered pairs of nodes. Very simple to implement.'''
    node_list = [x for x in adj_mat_dict] 
    pair_list = []
    for node1 in node_list:
        for node2 in node_list:
            if node1 != node2:
                pair_list.append((node1, node2))
    return pair_list

def AvgMinPathLen(adj_mat):
    '''This function takes an adjacency matrix as the input and finds the minimum
    path length for every pair of nodes. Then, it averages it out.'''    
    adj_mat_dict = AdjMat2Dict(adj_mat)
    pair_list = NodePairs(adj_mat_dict)
    min_path_lengths = []
    for pair in pair_list:
        min_path_lengths.append(MinPathLen(adj_mat_dict, pair[0], pair[1]))
    avg = np.mean(min_path_lengths)
    return round(avg,2)
    
'''A series of supporting functions to help in getting my final output and in checking.'''    
def NumEdges(graph_list):
    no_edges = []
    for matrix in graph_list:
        no_edges.append(np.sum(matrix))
    return no_edges

def AvgEdges(list_of_edge_lengths):
    a = np.mean(list_of_edge_lengths)
    return round(a, 2)
    
def EdgesRange(list_of_edge_lengths):
    return min(list_of_edge_lengths), max(list_of_edge_lengths)
    
'''Now that I have a valid list of graphs, I would like to calculate the 
number of valid paths in the graph for each graph in the list I have.
Then, I have to calculate the average number of paths.'''
def AvgMin(graph_list):
    scores = []
    for matrix in graph_list:
        scores.append(AvgMinPathLen(matrix))
    avg = np.mean(scores)
    return round(avg,2)

