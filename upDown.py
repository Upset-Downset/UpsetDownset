#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 11:15:10 2020

Last edited by Charlie: Aug 13 20:58 2020

@author: Charlie & Jamison

Let P be a finite poset with coloring c:P-> {-1,0,1} where 1 (resp. 0,-1) 
represent blue (resp. green, red). The Upset Downset game on P is the 
short partizan combinatorial game with the following possible moves: 
    - For any element x in P colored blue or green, Left (Up) may remove the 
    upset >x of x leaving the colored poset P - >x, 
    - and for x in P colored red or green, Right (Down) may remove the downset
    <x of x leaving the colored poset P - <x.
The first player unable to move loses. 

**Upset Downset is usually played on the Hasse diagram of P. The available 
moves for Up and Down are then:
    - Up may choose to remove any blue or green colored vertex on the Hasse 
    diagram of P along with all vertices connected to it by a path moving 
    strictly upward. 
    - Down may choose to remove any red or green colored vertex on the Hasse 
    diagram of P along with all vertices connected to it by a path moving 
    strictly downward.
The first player who cannot remove any vertices loses.
"""

import random
import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import HarryPlotter as hp

# Functions on posets: as of right now we use the networkx digraph to 
# represent a partially ordered set.
def upset(P, x):
    '''
    Returns the upset >x of x in P.
    '''
    upst = nx.descendants(P, x)
    upst.add(x)
    return upst

def downset(P, x):
    
    '''
    Returns the downset <x of of x in P.
    '''
    dwnst = nx.ancestors(P, x)
    dwnst.add(x)
    return dwnst

def coloredDual(P):
    '''
    Returns the colored dual of P. (If P has coloring c, then the colored 
    dual of P has coloring -c.)
    '''
    C = P.copy()
    for x in nx.nodes(C):
        C.nodes[x]['color'] = - P.nodes[x]['color']
    return nx.reverse(C) 

def disjointUnion(P,Q):
    '''
    Returns the disjoint union of P and Q.
    '''
    return nx.disjoint_union(P,Q)

def height(P, x):
        '''
        Returns the height of the element x. (The length of the longest chain 
        in the downset of x.)
        '''
        x_dwnst = downset(P, x)
        x_dwnst_sub_dag = nx.subgraph(P, x_dwnst)
        return nx.dag_longest_path_length(x_dwnst_sub_dag)
    
def hasse(P, pos=None):
    '''
    Plots the hasse diagram. Returns the position of the vertices.
    '''
    # vertical cooridnates
    vertical = {x: height(P,x) for x in nx.nodes(P)}
    # horizontal coordinates: if elements are not labelled by consecutive 
    # nonnegative integers starting from 0 we must force such a scaling.
    horizontal = {x: i for i,x in enumerate(nx.nodes(P))}
    # coordinate positions of vertices in hasse
    if pos == None:    
        pos = {x:(horizontal[x],vertical[x]) for x in nx.nodes(P)}
    # convert coloring to actual colors while grouping together elements 
    # of same color 
    colors = {'b':set(), 'g': set(), 'r': set()}
    for x in nx.nodes(P):
        if P.nodes[x]['color'] == 1:
            colors['b'].add(x)
        elif P.nodes[x]['color']  == 0:
            colors['g'].add(x)
        else:
            colors['r'].add(x)
    # create plot
    plt.figure(figsize = (12,12), frameon = False)
    # draw vertices
    nx.draw_networkx_nodes(P, pos, nodelist = colors['b'], node_color = 'b')
    nx.draw_networkx_nodes(P, pos, nodelist = colors['g'], node_color = 'g')
    nx.draw_networkx_nodes(P, pos, nodelist = colors['r'], node_color = 'r')
    # draw edges
    nx.draw_networkx_edges(P, pos, arrows = False,)
    #draw vertex labels
    nx.draw_networkx_labels(P, pos,)
    
    return pos
        
def randomPoset(n, colored = False):
    '''
    Returns a randomly generated poset with n elements lablelled 0,2,..., n-1. 
    Optionally colored, all green by default.
    '''
    # Initialize coloring.
    if colored == True:
        colors = [-1, 0, 1]
    else:
        colors = [0]
    #initialize empty DAG and return if no vertices.
    G = nx.DiGraph()
    if n == 0:
        return G
    # Add vertcies, color and return if only one vertex.
    for i in range(n):
        G.add_node(i, color = random.choice(colors))
    if n == 1:
        return G
    # Randomy choose the number of edges.
    e = random.randint(n, 2*n)
    #add edges. 
    while e > 0:
        #pick two distinct edges randomly
        v = random.randint(0, n-1)
        u = v
        while u == v:
            v = random.randint(0, n-1)
        # add edge from v to u.
        G.add_edge(v,u)
        #check if G is acyclic.
        if nx.is_directed_acyclic_graph(G):
            e -= 1
        else:
            G.remove_edge(v,u)
    # Transitively reduce.
    T = nx.transitive_reduction(G)
    # Color the vertices.
    for x in nx.nodes(T):
        T.nodes[x]['color'] = G.nodes[x]['color']
    return T
    
def completeBipartitePoset(m,n, colored = False):
    '''
    Returns the poset whose hasse diagram is goven by the complete 
    bipartite graph having n vertices (labelled 0,..., n-1) on the bottom 
    level and m vertices (labelled n,...,m+n-1) on the top level. Optionally 
    colored, all green by default and randomly otherwise
    '''
    # Initialize coloring.
    if colored == True:
        colors = [-1, 0, 1]
    else:
        colors = [0]
    P = nx.DiGraph()
    for i in range(n+m):
        P.add_node(i, color = random.choice(colors))
    for i in range(n):
        for j in range(n, n+m):
            P.add_edge(i,j)
    return P

def graphPoset(G):
    '''
    Returns the (colored) poset P derived from the (colored) graph G. P is 
    defined as follows: 
    - As a set P is the disjoint union of the Vertex set V and edge set E of G.
    - All vertices of G are incomparable.
    - No edge is less than any vertex.
    - Edge e is greater than vertex v if e is adjacent to v in G.
    '''
    g = nx.convert_node_labels_to_integers(G)
    n = nx.number_of_nodes(g)
    P = nx.DiGraph()
    for v in nx.nodes(g):
        P.add_node(v, color = g.nodes[v].get('color',0))
    for i, e in enumerate(nx.edges(g)):
        P.add_node(n+i, color = g.edges[e].get('color',0))
        P.add_edge(e[0], n+i)
        if e[0] != e[1]:
            P.add_edge(e[1], n+i)
    return P

# UpDown class structure
    
class UpDown(object):
    '''
    Abstract class for constructing an upset downset game from a directed 
    acyclic graph with vertices colored red, green or blue. 
    (I.e., a finite poset with elements colored red, greeen or blue.)
    '''
    def __init__(self, P):
        self.poset = P
        
    def getPoset(self):
        return self.poset
         
    def upOptions(self):
        '''
        Returns a dictionary of Ups options keyed by element.
        '''
        up_options = {}
        upsets = {}
        for x in nx.nodes(self.poset):
            if self.poset.nodes[x]['color'] in {0,1}:
                upsets[x] = upset(self.poset, x)
        for x in upsets:
            H = copy.deepcopy(self.poset)
            H.remove_nodes_from(upsets[x])
            up_options[x] = UpDown(H)
        return up_options

    def downOptions(self):
        '''
        Returns a dictionary of Downs options keyed by element.
        '''
        down_options = {}
        downsets = {}
        for x in nx.nodes(self.poset):
            if self.poset.nodes[x]['color'] in {-1,0}:
                downsets[x] = downset(self.poset, x)
        for x in downsets:
            H = copy.deepcopy(self.poset)
            H.remove_nodes_from(downsets[x])
            down_options[x] = UpDown(H)
        return down_options
    
    def outcome(self):
        '''
        Returns the outcome. 
        '''
        n = nx.number_of_nodes(self.poset)
        e = nx.number_of_edges(self.poset)    
        N, P, L, R = 'Next', 'Previous', 'Up', 'Down'
        colors = {x:self.poset.nodes[x]['color'] \
                  for x in nx.nodes(self.poset)}
        # Base cases for recursion
        colors_sum = sum(colors.values()) 
        if e == 0:
            if colors_sum == 0:
                if n%2 == 0:
                    out = P
                else:
                    out = N
            elif colors_sum > 0:
                out = L
            else:
                out = R
        # Recursively determine the ouctome.
        else:
            up_ops_otcms = {G.outcome() for G in self.upOptions().values()}
            dwn_ops_otcms = {G.outcome() for G in self.downOptions().values()}
            # first player to move wins
            if (P in up_ops_otcms or L in up_ops_otcms) and \
                (P in dwn_ops_otcms or R in dwn_ops_otcms):
                out = N
            # up wins no matterwho moves first
            elif (P in up_ops_otcms or L in up_ops_otcms) and \
                (P not in dwn_ops_otcms and R not in dwn_ops_otcms):
                out = L
            # down wins no matter who moves first
            elif (P in dwn_ops_otcms or R in dwn_ops_otcms) and \
                (P not in up_ops_otcms and L not in up_ops_otcms):
                out = R
            # second player to move wins
            elif (P not in up_ops_otcms and L not in up_ops_otcms) and \
                (P not in dwn_ops_otcms and R not in dwn_ops_otcms):
                out = P
        return out
    
    def __neg__(self):
        '''
        Returns the negative of the game.
        '''        
        return UpDown(coloredDual(self.poset))

    def __add__(self, other):
        '''
        Retruns the sum of the two games.
        '''
        return UpDown(disjointUnion(self.poset, other.poset))
    
    def __sub__(self, other):
        '''
        Retruns the difference of the two games.
        '''
        return self + (- other)
    
    def __eq__(self, other):
        '''
        Returns True if the two games are equal and False otherwise.
        '''
        return (self - other).outcome() == 'Previous'
    
    def __or__(self, other):
        '''
        Returns True if the two games are fuzzy (incomparable) and False 
        otherwise.
        '''
        return (self - other).outcome == 'Next'

    def __gt__(self, other):
        '''
        Returns True if the first game is greater than (better for Up) the 
        second game and False otherwise.
        '''
        return (self - other).outcome() == 'Up'
    
    def __lt__(self, other):
        '''
        Returns True if the first game is less than (better for down) the 
        second game and False otherwise.
        '''
        return (self - other).outcome() == 'Down'
            
    def gameboard(self):
        '''
        Plots the gameboard. Returns position of the vertices.
        '''
        return hasse(self.poset)
    
    def get_height(self):
        '''
        Returns the number of levels -1

        '''
        x = 0
        for i in self.poset.nodes():
            if x < height(self.poset, i):
                x = height(self.poset, i)
        return x
        
    def play(self):
        '''
        Plays a game of UpDown. Sets the position to be the starting position
        of the gameboard.
        '''
        plt.clf()
        plt.ioff()
        players = ['Up', 'Down']
        first = input("Which player will start, 'Up' or 'Down'? ")       
        while first not in players:
            first = input("Please choose 'Up' or 'Down'! ")
        players.remove(first)
        second = players[0]
        if first == 'Up':
            first_colors, second_colors = {0,1}, {-1,0}
        else:
            first_colors, second_colors = {-1,0}, {0,1}
                        
        i = 0
        fig_info = hp.harry_plotter(self)
        plt.close(1)
        fig_info[0].suptitle("GAME")
        plt.pause(0.01)
        while self.poset.nodes():           
            first_options = list(filter(lambda x : self.poset.nodes[x]['color'] in first_colors, \
                  self.poset.nodes()))
            second_options = list(filter(lambda x : self.poset.nodes[x]['color'] in second_colors, \
                  self.poset.nodes()))
                
            if not first_options:
                i=0
                break
                
            if not second_options:
                i=1
                break
                
            if first == "Up":
                if i % 2 == 0:
                    u = int(input(first + ", choose a Blue or Green node: "))
                    while not (u in first_options):
                        print(u, " is not a valid choice.")
                        u = int(input(first + ", choose a Blue or Green node: ")) 
                    sub_game = self.upOptions()[u]
                    hp.figure_subgraph(sub_game, fig_info)
                    self = sub_game
                    plt.pause(0.01)
                    
                #second players turn
                else:
                    u = int(input(second + ", choose a Red or Green node: "))
                    while not (u in second_options):
                        print(u, " is not a valid choice.")
                        u = int(input(second + ", choose a valid Red or Green node: "))
                    sub_game = self.downOptions()[u]
                    hp.figure_subgraph(sub_game, fig_info)
                    self = sub_game
                    plt.pause(0.01)
                    
            else:
                if i % 2 == 0:
                    u = int(input(first + ", choose a Red or Green node: "))
                    while not (u in first_options):
                        print(u, " is not a valid choice.")
                        u = int(input(first + ", choose a Red or Green node: ")) 
                    sub_game = self.downOptions()[u]
                    hp.figure_subgraph(sub_game, fig_info)
                    self = sub_game
                    plt.pause(0.01)
                #second players turn
                else:
                    u = int(input(second + ", choose a Blue or Green node: "))
                    while not (u in second_options):
                        print(u, " is not a valid choice.")
                        u = int(input(second + ", choose a valid Blue or Green node: "))
                    sub_game = self.upOptions()[u]
                    hp.figure_subgraph(sub_game, fig_info)
                    self = sub_game
                    plt.pause(0.01)
            i += 1
            
        if i % 2 == 0:
            print(second + ", wins!")
            fig_info[0].suptitle(second + ", wins!")
        else:
            print(first + ", wins!")
            fig_info[0].suptitle(first + ", wins!")
        plt.pause(0.01)
# Subclasses of UpDown      
class RandomUpDown(UpDown):
    '''
    Constructor for a randomly generated game of UpDown.
    '''
    def __init__(self, n, colored = False):
        '''
        Initializes a randomly generated game of Upset Downset on n verticies. 
        Optionally colored, all green by default.
        '''
        P = randomPoset(n, colored = colored)
        UpDown.__init__(self, P)
        
class CompleteBipartiteUpDown(UpDown):
    '''
    Constructor for a game of Upset Downset played on a poset whose hasse 
    diagram is a (horizontally oriented) complete bipartite graph.
    '''
    def __init__(self, m, n, colored = False):
        '''
        Initializes a complete bipartite game of Upset Downset on m+n 
        verticies. Optionally colored, all green by default and randomly 
        otherwise.
        '''
        P = completeBipartitePoset(m, n, colored = colored)
        UpDown.__init__(self, P)
        
class GraphUpDown(UpDown):
    '''
    Abstract class for constructing an Upset Downset game played on a colored
    (blue, green, red) graph G. Here the poset P in which the game is played 
    on is derived from G as follows: 
        - As a set P is the disjoint union of the Vertex set V and edge set E 
        of G.
        - All vertices of G are incomparable.
        - No edge is less than any vertex.
        - The edge e is gretaer than a vertex v if e is adjacent to v in G.
    
    **The Graph Upset Downset game is played on G, not on the hasse diagram of 
    P.
    
    '''
    def __init__(self, G, colored = False):
        '''
        Initializes a game of Graph Upset Downset on the graph G. Optionally 
        colored, all green by default.
        '''
        self.graph = G
        P = graphPoset(self.graph)
        UpDown.__init__(self, P)
    
    def getGraph(self):
        return self.graph
    
    def drawGraph(self):  #UPDATE TO DRAW LOOPS
        v_colors = {}
        e_colors = {}
        for v in nx.nodes(self.graph):
            if self.graph.nodes[v]['color'] == 0:
                c = 'g'
            elif self.graph.nodes[v]['color'] == 1:
                c = 'b'
            else:
                c = 'r'
            v_colors[v] = c
        for e in nx.edges(self.graph):
            if self.graph.edges[e]['color'] == 0:
                c = 'g'
            elif self.graph.edges[e]['color'] == 1:
                c = 'b'
            else:
                c = 'r'
            e_colors[e] = c
        return nx.draw(self.graph, with_labels = True, \
                       node_color = v_colors.values(), \
                           edge_color = e_colors.values())
            
    def gameboard(self):
        '''
        Plots the gameboard.
        '''
        self.drawGraph()
        
class RandomGraphUpDown(GraphUpDown):
    '''
    Constructor for a randomly generated game of Graph Upset Downset.
    '''
    def __init__(self, m, n, colored = False):
        '''
        Initializes a randomly generated game of Graph Upset Downset on a 
        graph with m vertices and n edges. Optionally colored, all green by 
        default and randomly otherwise.
        '''
        G = nx.gnm_random_graph(m,n)
        # Initialize coloring.
        if colored == True:
            colors = [-1, 0, 1]
        else:
            colors = [0]
        for v in nx.nodes(G):
            G.nodes[v]['color'] = random.choice(colors)
        for e in nx.edges(G):
            G.edges[e]['color'] = random.choice(colors)
        GraphUpDown.__init__(self, G)
