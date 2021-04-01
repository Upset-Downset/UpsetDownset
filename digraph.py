"""
@author: Charles Petersen and Jamison Barsotti
"""
import copy
import collections 

def relabel(G, relabel_map):
    ''' Returns a copy of the directed graph 'G' with nodes relabelled
    according to 'relabel_map'. 
    
    Parameters  
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacency list keyed 
        by node.)
    
    relabel_map : dict
        new node labels keyed by old node labels. 
        
    Returns
    -------
    graph: dict
        adjacency representation of the directed graph 'G' with nodes 
        relabelled by 'relabel_map'.
    '''
    graph = {}
    for v in G:
        graph[relabel_map[v]] = []
        for u in G[v]:
            graph[relabel_map[v]].append(relabel_map[u])
    return graph
 
def subgraph(G, nodes):
    ''' Returns the subgraph of the directed graph 'G' on 'nodes'.
    
    Parameters
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacency list keyed 
        by node.)
        
    nodes : list
        subset of nodes of 'G' on which the subgraph is to be defined.
        
    Returns
    -------
    dict
        adjacency representation of the subgraph of 'G' on 'nodes'. (Adjacency 
        lists keyed by node.)       

    '''
    sub_graph = {}
    for v in nodes:
        sub_graph[v] = []
        for u in G[v]:
            if u in nodes:
                sub_graph[v].append(u)              
    return sub_graph

def reverse(G):
    ''' Returns the reverse of the directed graph 'G'.
    
    Parameters
    ----------
    G : dict 
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)

    Returns
    -------
    dict
        adjacency representation of the reverse of 'G'. (Adjacency lists keyed 
        by node.)
        
    '''
    reverse = {v:[] for v in G}
    for v in G:
        for u in G[v]:
            reverse[u].append(v)
    return reverse

def edge_list(G):
    '''Returns all edges of the directed graph 'G'.
    
    Parameters
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacency lists 
        keyed by node.)

    Returns
    -------
    list
        lists of ordered-pairs denoting the edges of G. A pair (u,v)
        correpsonds to the edge u --> v in 'G'.

    '''
    edges = []
    for u in G:
        for v in G[u]:
            edges.append((u,v))   
    return edges

def number_of_edges(G):
    ''' Returns the number of edges in the directed graph 'G'.

    Parameters
    ----------
    G : dict 
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)

    Returns
    -------
    int
        the number of edges in 'G'.

    '''
    return sum(len(G[v]) for v in G)

def sinks(G):
    ''' Returns all sink nodes in the directed graph 'G'.
    
    Parameters
    ----------
    G : dict 
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)
        
    Returns
    -------
    list
        all sinks of 'G'.  (A node is a sink if it has no outgoing edges.)

    '''
    return list(filter(lambda x : G[x] == [], G.keys())) 

def sources(G):
    ''' Returns all source nodes in the directed graph 'G'.
    
    Parameters
    ----------
    G : dict 
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)
        
    Returns
    -------
    list
        all sources of 'G'. (A node is a source if it has no incoming edges.)
        
    '''
    return sinks(reverse(G))

def descendants(G, source):
    ''' Returns all nodes reachable from 'source' node in the directed 
    graph 'G'.
    
    Parameters
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)
        
    source : int (nonnegative)
        node of 'G'. 

    Returns
    -------
    list
        all nodes reachbale from 'source' in 'G'.
    
    Reference:
        Introduction to Algotithms: Chapter 22, Thomas H. Cormen, 
        Charles E. Leiserson, Ronald L. Rivest, Cliffors Stein, 3rd Edition, 
        MIT Press, 2009. 
        
    '''
    def dfs_visit(G,v):
        visited[v] = 0
        reachable_from_source.append(v)
        for u in G[v]:
            if visited[u] == -1:
                dfs_visit(G,u)
        visited[v] = 1
        
    reachable_from_source = []
    visited = {v:-1 for v in G}
    for v in G[source]:
        if visited[v] == -1:
            dfs_visit(G,v)
            
    return reachable_from_source
   
def ancestors(G, source):
    ''' Returns all nodes having a path to 'source' node in the directed 
    graph 'G'.
    
    Parameters
    ----------
    G : dict
        adjacencyrepresentation of a directed graph. (Adjacency lists keyed 
        by node.)
        
    source : int (nonnegative)
        node of 'G'.

    Returns
    -------
    list
        all nodes having a path to 'source' in 'G'.

    '''
    return descendants(reverse(G), source)

def is_acyclic(G):
    ''' Returns wether the directed graph G is acyclic.
    
    Parameters
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)

    Returns
    -------
    bool
        True if 'G' is acyclic and False otherwise.
    
    Reference:
        Introduction to Algotithms: Chapter 22, Thomas H. Cormen, 
        Charles E. Leiserson, Ronald L. Rivest, Cliffors Stein, 3rd Edition, 
        MIT Press, 2009.

    '''
    def dfs_visit(G,v):
        visited[v] = 0
        for u in G[v]:
            # if we encounter a node again before we've finished exploring it's 
            # adjacecny list there must be a cycle!
            if visited[u] == 0:
                return False
            if visited[u] == -1:
                if not dfs_visit(G,u):
                    return False
        visited[v] = 1
        return True

    visited = {v:-1 for v in G}
    for v in G:
        if visited[v] == -1:
            if not dfs_visit(G,v):
                return False
            
    return True

def topological_sort(G, reverse = False):
    ''' Returns a topological ordering of the nodes in the directed acyclic
    graph 'G'. (It is assumed that 'G' is acyclic, we do not check.)

    Parameters
    ----------
    G : dict
        adjacency representation of a directed acyclic graph. (Adjacency lists 
        keyed by node.)
        
    reverse : bool, optional
        if True, the reverse of the topological ordering will be returned. 
        The default is False.

    Returns
    -------
    list
        a topological ordering of the nodes in 'G'. 
    
    Reference:
        Introduction to Algotithms: Chapter 22, Thomas H. Cormen, 
        Charles E. Leiserson, Ronald L. Rivest, Cliffors Stein, 3rd Edition, 
        MIT Press, 2009.

    '''
    def dfs_visit(G,v):
        visited[v] = 0
        for u in G[v]:
            if visited[u] == -1:
                dfs_visit(G,u)  
        # Done with v, add it to ordering.
        visited[v] = 1
        order.append(v)
        
    order = []
    visited = {v:-1 for v in G}
    for v in G:
        if visited[v] == -1:
            dfs_visit(G,v)
    # Since we were appending finished nodes to the list we need to reverse.
    if not reverse:
        order.reverse()
        
    return order

def transitive_closure(G):
    ''' Returns the transitive closure of the directed graph 'G'.
    
    Parameters
    ----------
    G : dict
        adjacency representation of a directed graph. (Adjacency lists keyed 
        by node.)

    Returns
    -------
    dict
        adjacency representation of the transitive closure of 'G'. (Adjacency 
        lists keyed by node.)
        
    Reference:
        https://en.wikipedia.org/wiki/Transitive_closure
         
    '''
    transitive_closure = {}
    for v in G:
        transitive_closure[v] = descendants(G,v)
    return transitive_closure

def transitive_reduction(G):
    ''' Returns the transitive reduction of the directed acyclic** graph 'G'. 
    (It is assumed that 'G' is acyclic, we do not check.)
    
    Parameters.
    ----------
    G : dict
        adjacency representation of a directed acyclic graph. (Adjacency lists 
        keyed by node.)

    Returns
    -------
    dict
        adjacency representation of the transitive reduction of 'G'. (Adjacency 
        lists keyed by node.
        
    References:
        - https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/dag.html
        - https://en.wikipedia.org/wiki/Transitive_reduction
        
        
    '''                 
    transitive_reduction = {}
    # update descendants of each node in G to dict as they are needed.
    des = {}
    for v in G:
        # nodes adjacent to v in transitive reduction.
        adj_v = set(G[v])
        # loop over nodes adjacent to v in G removing their descendants from 
        # v's adjaceny list in transitive reduction
        for u in G[v]:
            # if u has already been removed from v's adjaceny list in 
            # transitive reduction, then so have u's descendants in G.
            if u in adj_v:
                if u not in des:
                    des[u] = set(descendants(G,u))
                adj_v -= des[u]
        transitive_reduction[v] = list(adj_v)
        
    return transitive_reduction

def longest_path_lengths(G, direction='outgoing'):
    ''' Returns the length of the longest path outgoing (optionally, 
    incoming)  each node in the directed acyclic graph 'G'. (It is assumed 
    that 'G' is acyclic, we do not check.)
    
    Parameters
    ----------
    G : dict
        adjacency representation of a directed acyclic graph. (Adjacency lists 
        keyed by node.)
        
    direction : str, optional
         If 'outgoing', the length of the longest path starting at each node 
         will be computed. If 'incoming', the length of the longest path 
         ending at each node will be computed.

    Returns
    ------- 
    dict
        lengths of the longest outgoing (optionally incoming) paths in 'G' 
        keyed by node.
    
    Reference: 
        - https://en.wikipedia.org/wiki/Longest_path_problem#Acyclic_graphs_and_critical_paths

    '''
    # Store maximum path length computed for each node
    max_path_lengths = {v:0 for v in G}
    # if we want incoming paths, then reverse the graph.
    if direction == 'incoming':
        G = reverse(G)
    reverse_linear_order = topological_sort(G, reverse = True)  
    # loop over nodes in reverse topological order, updating maximum length of 
    # each path as we go.
    for v in reverse_linear_order:
        MAX = 0
        for u in G[v]:
            if max_path_lengths[u] + 1 > MAX:
               MAX = max_path_lengths[u] + 1
        max_path_lengths[v] = MAX  
        
    return max_path_lengths

def connected_components(G):
    ''' Returns the nodes in each component of the undirected graphs
    underlying the directed graph 'G"."

    Parameters
    ----------
    G : dict
        adjacency representation of a directed acyclic graph. (Adjacency lists 
        keyed by node.)

    Returns
    -------
    list
        lists of nodes in each connected compnent of the undirected graph 
        underlying 'G'.
    '''
    def dfs_visit(G,v):
        visited[v] = 0
        for u in undirected[v]:
            if visited[u] == -1:
                v_component.add(u)
                dfs_visit(undirected,u)  
        visited[v] = 1
        
    components = []
    # if no nodes return empty list
    if len(G) == 0:
        return components
    # get undirected version of G.
    undirected = copy.deepcopy(G)
    rev = reverse(G)
    for v in undirected:
        undirected[v].extend(rev[v])
    # to be updated as we discover each component.
    nodes = set(G)
    visited = {v:-1 for v in G}
    v = nodes.pop()
    # dfs to find all components of G
    while True:
        v_component = {v}
        dfs_visit(undirected,v)
        components.append(v_component)
        nodes = nodes - v_component
        if nodes:
            v = nodes.pop()
        else: 
            break
        
    return components


def hasse_layout(G):
    ''' Returns the xy-coordinates of each node in a Hasse diagram plot layout
    of the directed acyclic graph 'G'. In particular, all arrows (edges) 
    should be pointing up. (It is assumed that 'G' is acyclic, we do not check.)

    Parameters
    ----------
    G : dict
        adjacency representation of a directed acyclic** graph. (Adjacency lists 
        keyed by node.)

    Returns
    -------
    dict
        xy-coordinates keyed by node:
            - the y-coordinate of each node is assigned by the level of 
            the node in 'G'. The level of a node N  in 'G' is the 
            the length of the longest path in 'G' which ends at N.
            - the x-coordinate of each node is assigned in such a way that
            for each component of 'G' each level for that component 
            is evenly spaced horizontally with previous level.
    
    NOTE: At times this layout can return an unwanted overlap between edges 
    and nodes. However, it works pretty well most of the time.
    
    Reference: 
        - https://en.wikipedia.org/wiki/Hasse_diagram
    '''
    # hasse diagram coordinates of nodes: coordinate keyed by node
    hasse_pos = {}
    components = connected_components(G)
    # loop over components setting coordinates for each component. At each 
    # iteration keep track of the largest x coordinate for use in spacing the
    # next component.
    prev_comp_max_x = 0
    for nodes_list in components:
        comp = subgraph(G,nodes_list)
        # level of each node in current component: level keyed by node
        levels_dict = longest_path_lengths(comp, direction = 'incoming')
        # level set decomp of current component: node lists keyed by level
        level_sets = collections.defaultdict(list)
        for key, val in sorted(levels_dict.items()):
            level_sets[val].append(key)
        # loop over all levels in current component, starting from bottom and 
        # moving up level by level. Initialize coordinates 
        # for all nodes in the current level. At each iteration we keep track
        # of the right most x coordinate and size of previous level
        widest_level_size = len(max(level_sets.values(), key=len))
        bottom_level_size = len(level_sets[0])
        x_gap = (widest_level_size - bottom_level_size)/2
        prev_level_left_x = prev_comp_max_x + x_gap + 1
        prev_level_size = 0
        for l in sorted(level_sets):
            level_size = len(level_sets[l])
            # we shift the nodes of current level by half the signed difference
            # between the size of the previous level and the current level. 
            # This evenly centers the levels on top of one another.
            if prev_level_size == 0:
                shift = 0
            else:
                shift = (prev_level_size - level_size)/2
            level_left_x = prev_level_left_x + shift
            x_coords = [level_left_x + i for i in range(level_size)]
            hasse_pos.update({node: (x,l) for node, x in \
                        zip(level_sets[l], x_coords)}) 
            # updates for next component and next level, respectively.
            prev_comp_max_x = max(prev_comp_max_x, level_left_x + level_size-1)
            prev_level_left_x = level_left_x
            prev_level_size = level_size  
            
    return hasse_pos
        
