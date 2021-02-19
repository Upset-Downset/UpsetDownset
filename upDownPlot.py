"""
@author: Charles Petersen and Jamison Barsotti
"""
import digraph    
import matplotlib.pyplot as plt

def color(c):
    ''' Assigns an RGB color value to 'c'.

    Parameters
    ----------
    c : int
        1, 0 or -1 (blue resp. green, red).

    Returns
    -------
    RGB color value
       RGB color value corresponding to 'c'.

    '''
    colordict = {-1 : '#F05252', 0 : '#09D365', 1 : '#5284F0'}
    return colordict[c]                           

class UpDownPlot(object):
    '''Abstract class for constructing and mutating a visualization of an 
    upset-downset game.
    '''

    def __init__(self, game, marker='o'):               
        '''
        Plots the  Hasse diagram (transitively reduced directed 
        acyclic graph with all edges pointing up) underlying the upset-downset 
        game, 'game'. The node positions in the plot are determined according
        to the 'hasse_layout' function in the digraph module.
        
        ** Besides simply plotting a game of upset-downset, when interactively 
        playing a game this class allows for the easy mutation of the plot so 
        as to keep edge and vertex positions consistent throughout play. (See 
        the leave_subgraph() method.)

        Parameters
        ----------
        game : UpDown
            a game of upset-downset
    
        Returns
        -------
        list
            inormation about the figure: 
            [0] fig : matplotlib.figure
            [1] fig_edges : dict, figure edges, keyed by edge labels.
            [2] fig_vertices : dict, figure vertices keyed by labels.
            [3] fig_vertex_labesl : dict, figure vertex labels keyed by 
            vertex labels 
        '''
        
        # get the dag, coloring, node and edge lists .     
        dag = game.dag            
        colors = game.coloring
        nodes = game.dag.keys()
        edges = digraph.edge_list(dag)
        pos = digraph.hasse_layout(dag)             
            
        # set up the figure and an axes. The empty dicts
        # will be filled by pointers to the lines, vertices 
        # and labels of the figure, so pieces can be removed with ease.
        fig = plt.figure()
        ax = fig.add_subplot()
        fig_edges = {}
        fig_vertices = {}
        fig_vertex_labels = {}
    
        for e in edges:
            # add a line segment to the figure connecting 
            # endpoints of edge
            ax.plot([pos[e[0]][0],pos[e[1]][0]],
                    [pos[e[0]][1],pos[e[1]][1]], color='000000')  
            # update fig_edges with a pointer to the line segment 
            # in the figure.
            fig_edges[(e[0],e[1])] = ax.lines[-1]
                            
        for n in nodes:
            # get the color of the node
            c = color(colors[n])     
            # add a point to the figure with color c.
            ax.plot(pos[n][0],pos[n][1], marker=marker, color=c, markersize=18)
            # add a label to the vertex
            ax.annotate(str(n), xy=(pos[n][0],pos[n][1]))
            # update fig_vertices and fig_vertex_labels with
            # pointers to the new additions.
            fig_vertices[n] = ax.lines[-1]
            fig_vertex_labels[n] = ax.texts[-1]
        
        # don't plot the x or y axes.
        plt.axis('off')
         
        self.figure = fig
        self.figure_edges = fig_edges
        self.figure_vertices = fig_vertices
        self.figure_vertex_labels = fig_vertex_labels


    def leave_subgraph_fig(self, sub_game):
        ''' Mutates 'self' to only contain edge, vertice and vertex label
        information of 'sub_game' leaving the remaining figure information 
        as it was.
        
        Parameters
        ----------
        sub_game : UpDown
            an option of the upset-downset game whose figure is given 
            by 'self'.            
    
        Returns
        -------
        None
        
        '''
        
        # get the nodes and edges of the sub game
        dag = sub_game.dag            
        nodes = sub_game.dag.keys()                      
        edges = digraph.edge_list(dag)
        
        # initialize dicts to store  sub game figure info
        sub_fig_edges = {}
        sub_fig_vertices = {}
        sub_fig_vertex_labels = {}
        
        # collect the sub game figure info
        for e in edges:
            sub_fig_edges[e] = self.figure_edges.pop(e)
        for n in nodes:
            sub_fig_vertices[n] = self.figure_vertices.pop(n)
            sub_fig_vertex_labels[n] = self.figure_vertex_labels.pop(n)
            
        # remove the figure info not in the sub game
        for e in self.figure_edges:
            self.figure_edges[e].remove()
        for v in self.figure_vertices:
            self.figure_vertices[v].remove()
            self.figure_vertex_labels[v].remove()
        
        # mutate the figure info to match whats left in the sub game
        self.figure_edges = sub_fig_edges
        self.figure_vertices = sub_fig_vertices
        self.figure_vertex_labels = sub_fig_vertex_labels
    
    def show(self):
        ''' Plots the game.
        Returns
        -------
        None.

        '''
        self.figure.show()
        
    def close(self):
        ''' Closes the game plot.
        
        Returns
        -------
        None.

        '''
        plt.close(self.figure)
        
