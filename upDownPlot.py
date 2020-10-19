#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 08:13:51 2020

@author: jamison
"""
import upDown as uD
import matplotlib.pyplot as plt
from dagUtility import *

def up_down_plot(UD, marker='o'):
    """
    Plots the underlying dag of an instance of the class UpDown.
    A node in the graph is plotted at the point (n,l), where n is the
    integer label of the node and l is its level. Since node --> level
    preserves the order of cover relations, this presentation will
    visually preserve up-sets and down-sets.
    

    Parameters
    ----------
    UD : UpDown
        UD is an instance of class UpDown from the module upDown. 
        We use UD to extract the following information.
        0) node labels
        1) node colors
        2) edge list 
        3) node levels.

    Returns
    -------
    list
        inormation about the figure
        [0] fig : matplotlib.figure
        [1] fig_edges : dictionary of figure edges with edge labels as keys
        [2] fig_vertices : dictionary of figure vertices with vertex labels as keys
        [3] fig_vertex_labesl : dictionary of figure vertex labels
                                with vertex labels as keys.
    """
    #define a color assignment function.
    def color(n):
        colordict = {-1 : '#F05252', 0 : '#09D365', 1 : '#5284F0'}
        return colordict[n]
    
    #get the cover relations, color dictionary, node and edge lists.
    #get levels for each node.
    covers = UD._cover_relations
    colors = UD._coloring_map
    nodes = list(covers.keys())
    edges = edge_list(covers)
    levels = UD.levels()
    
    #Set up the figure and an axes.
    #The empty dictionaries will be filled pointers to the lines and labels
    #of the figure, so pieces can be removed with ease.
    fig = plt.figure()
    ax = fig.add_subplot()
    fig_edges = {}
    fig_vertices = {}
    fig_vertex_labels = {}

    #Iterate over the edge set
    for i in edges:
            #add a black line segment to the figure connecting the vertices i[0] and i[1].
            ax.plot([i[0],i[1]],[levels[i[0]],levels[i[1]]], color='000000')  
            #update the dictionary fig_edges with a pointer to the line segment in the figure.
            fig_edges[(i[0],i[1])] = ax.lines[-1]
                    
    #Iterate over the node set
    for i in nodes:
        #get the color of the node
        c = color(colors[i])
        
        #add a point to the figure corresponding to (i, levels[i]) with color c.
        ax.plot(i,levels[i], marker=marker, color=c, markersize=18)
        #add a label to the vertex
        ax.annotate(str(i), xy=(i-.025,levels[i]-.025))
        #update the fic_vertices and fig_vertex_labels dictionaries with
        #pointers to the new additions.
        fig_vertices[i] = ax.lines[-1]
        fig_vertex_labels[i] = ax.texts[-1]
    
    #don't plot the x or y axes.
    plt.axis('off')
     
    return [fig, fig_edges, fig_vertices, fig_vertex_labels]


def leave_subgraph_fig(H, fig_info):
    """
    If G is an UpDown, fig_info is the information about the matplotlib figure of G, 
    and H is an Updwon whose underlying graph is a subgraph of G, then this function removes
    any part of the matplotlib figure of G that is not also in H.
    Additionally, the function returns, the figure information of H.
    

    Parameters
    ----------
    H : UpDown
        A subgraph of G (G is an UpDown whose figure has been plotted)
    
    fig_info : list
        A list containing information about the figure of G.
        [0] fig : matplotlib.figure
        [1] fig_edges : dictionary of figure edges with edge labels as keys
        [2] fig_vertices : dictionary of figure vertices with vertex labels as keys
        [3] fig_vertex_labesl : dictionary of figure vertex labels
                                with vertex labels as keys.
            

    Returns
    -------
    None
    
    Effect
    -------
    Mutates fig_info to only contain the information of the subgraph H.
    """
    
    #Get the nodes and edges of H.
    covers = H._cover_relations
    nodes = list(covers.keys())
    edges = edge_list(covers)
    
    #Get dictionaries for the information of the figure of 
    #the subgraph H in G.
    sub_fig_edges = {}
    sub_fig_vertices = {}
    sub_fig_vertex_labels = {}
    
    #Collect the figure information of the subgraph.
    for i in edges:
        sub_fig_edges[i] = fig_info[1].pop(i)
    for j in nodes:
        sub_fig_vertices[j] = fig_info[2].pop(j)
        sub_fig_vertex_labels[j] = fig_info[3].pop(j)
        
    #Remove the information not in the subgraph from the figure.
    for i in fig_info[1]:
        fig_info[1][i].remove()
    for j in fig_info[2]:
        fig_info[2][j].remove()
        fig_info[3][j].remove()
    
    #Change the figure information to what's left in the figure.
    fig_info[1] = sub_fig_edges
    fig_info[2] = sub_fig_vertices
    fig_info[3] = sub_fig_vertex_labels