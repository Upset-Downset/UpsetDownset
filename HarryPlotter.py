#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 08:13:51 2020

@author: jamison
"""
import upDown as uD
import matplotlib.pyplot as plt

def harry_plotter(UD, marker='o'):
    """
    A plotting wizard for a DAG (directed acyclic graph)

    Parameters
    ----------
    UD : UpDown
    UD is an instance of class UpDown from the module upDown. 
    We use UD to extract the following information.
    
    UD --> dag : directed acyclic graph information with nodes set labelled sequentially
        from 0. Each node includes
        0) node label
        1) node height
        2) node color
        3) node (immediate) downset as a tuple. (contains None if none)

    Returns
    -------
    Tuple of inormation about the figure
    0) fig : matplotlib.figure
    1) fig_edges : dictionary of figure edges with edge labels as keys
    2) fig_vertices : dictionary of figure vertices with vertex labels as keys
    3) fig_vertex_labesl : dictionary of figure vertex labels
                            with vertex labels as keys.
    """
    def color(n):
        colordict = {-1 : '#F05252', 0 : '#09D365', 1 : '#5284F0'}
        return colordict[n]
    DAG = UD.poset
    dag = []
    nodes = list(DAG.nodes())
    edges = list(DAG.edges())
    
    for i in nodes:
        dag.append([i, uD.height(DAG, i), color(DAG.nodes[i]['color']), []])
    for j in edges:
        dag[nodes.index(j[1])][3].append(j[0])

    fig = plt.figure()
    ax = fig.add_subplot()
    fig_edges = {}
    fig_vertices = {}
    fig_vertex_labels = {}

    for i in dag:
        if i[3] != None:
            for j in i[3]:
                ax.plot([i[0],dag[nodes.index(j)][0]],\
                        [i[1],dag[nodes.index(j)][1]], color='000000')  
                fig_edges[(dag[j][0]), i[0]] = ax.lines[-1]
                    
    for i in dag:
        ax.plot(i[0],i[1], marker=marker, color=i[2], markersize=18)
        ax.annotate(str(i[0]), xy=(i[0]-.025,i[1]-.025))
        fig_vertices[i[0]] = ax.lines[-1]
        fig_vertex_labels[i[0]] = ax.texts[-1]
    plt.axis('off')
     
    return [fig, fig_edges, fig_vertices, fig_vertex_labels]


def figure_subgraph(sub_graph, fig_info):
    """
    If G is a DAG with figure information given in
    fig_info, and H is a subgraph (sub_graph) of G, function
    will draw the subgraph and return relevant dictionaries
    of info
    

    Parameters
    ----------
    sub_graph : Instance of UpDown from module upDown.
        sub_graph of the figure whose information is 
        given in fig_info.
    
    fig_info : tuple
        A tuple containing the information provided by
        running parent graph G through harry_plotter

    Returns
    -------
    Stuff.

    """
    DAG = sub_graph.poset
    nodes = list(DAG.nodes())
    edges = list(DAG.edges())
    sub_fig_edges = {}
    sub_fig_vertices = {}
    sub_fig_vertex_labels = {}
    
    for i in edges:
        sub_fig_edges[i] = fig_info[1].pop(i)
    for j in nodes:
        sub_fig_vertices[j] = fig_info[2].pop(j)
        sub_fig_vertex_labels[j] = fig_info[3].pop(j)
        
    for i in fig_info[1]:
        fig_info[1][i].remove()
    for j in fig_info[2]:
        fig_info[2][j].remove()
        fig_info[3][j].remove()
    
    fig_info[1] = sub_fig_edges
    fig_info[2] = sub_fig_vertices
    fig_info[3] = sub_fig_vertex_labels
