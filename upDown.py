#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""

import dagUtility as dag
import upDownPlot as udp                                                       
import matplotlib.pyplot as plt
import random
import time

class UpDown(object):
    ''' Abstract class for construction of an upset-downset game from a 
    blue-green-red partially ordered set (poset). 
    '''
    def __init__(self, relations, coloring = None, coordinates = None, covers = False):
        '''
        Parameters
        ----------
        relations : dict
            each key is an element in the underlying poset w/ corresponding value 
            being a list of elements greater than the key. The 'relations' may 
            contain some or all of the relational data of the poset with a few 
            caveats:
                - The dictionary must contain a key for each element in the poset. 
                  (The dict value of a maximal element should be an empty list.)
                - The elements are to be labelled by nonnegative integers.
                - At a  minimum all cover relations must be present to obtain 
                the intended poset.
                - Reflexivity is assumed. Do not provide the reflexive relations.
                (Refleive relations will give loops in the corresponding DAG.)
                - Anti-symmetry is checked via acylicity of the corresponding DAG
                - Transitivity is forced via reachability in the corresponding DAG.
                (In particular, the poset returned is the transitive closure of the 
                provided 'relations'.)
            **We identify the underlying poset with its Hasse diagram (transitoively 
            reduced directed acyclic graph, DAG) as usual:
                    nodes of DAG <----> elements of poset 
                    reachability in DAG <----> relation in poset 
                    (b reahcbale from a in dag <----> a<=b in poset) 
        coloring: dict, optional
            coloring map on elements in underlying poset. color keyed by element
            where 1 (resp. 0, -1) represents blue (resp. green, red). Default in None,
            in this case all elements will be colored green.
        coordinates: dict, optional
            coordinates of vertices in Hasse diagram of unerlying poset (the 'gameboard')
            keyed by corresponding poset elements: tuples should be of the 
            following form: (horizontal coordinate, vertical coordinate). The 
            default is 'None'. In this case horizontal coordinates are determined 
            by the label of the element and the vertical coordinate is determined by 
            the level of the element. (See the levels() method.) ***It is not 
            recommended to alter the vertical coordinate expect by scaling.       
        covers : bool, optional
            set to 'True' if the given 'relations' are the (upper) covering 
            relations on the underlying poset. The default is False.
        '''
        # Check for cycles in Hasse diagram
        assert dag.is_acyclic(relations), \
            'Check the relations. There is a cycle present in the Hasse diagram.'
        # Set relations: if not given exactly the covering relations take 
        # the transitive reduction.
        if covers is False:
            self.cover_relations = dag.transitive_reduction(relations)
        else:
            self.cover_relations = relations
        # List of elements in poset
        self.elements = list(self.cover_relations)
        # Set coloring, if none given color all elements green
        if coloring is None:
            self.coloring = {x:0 for x in self.elements}
        else:
            self.coloring = coloring
        # Set the coordinates of the elements in the Hasse diagram. 
        if coordinates is None:
            self.hasse_coordinates = {x:(x, self.levels()[x]) for x in self.elements}
        else:
            self.hasse_coordinates = coordinates
        
##############################################################################
######################### POSET ##############################################
##############################################################################
    
    def __len__(self):
        ''' Returns the cardinality of the underlying poset. Overloads the 'len'
        operator.

        Returns
        -------
        int
            number of elements in the underlying poset.

        '''
        return len(self.elements)
    
    def levels(self):
        ''' Returns the level of each element in the Hasse diagram of the 
        underlying poset. The level of an elemnt is the length of the longest 
        path in the elements downset.

        Returns
        -------
        dict
            levels keyed by elements.

        '''
        return dag.longest_path_lengths(self.cover_relations, direction = 'incoming')
        
    def height(self):
        ''' Returns the height of the underlying poset. 
        
        Returns
        -------
        int
            the maximum level in the underlying poset. (** See levels().)

        '''
        return max(self.levels().values())
    
    def maximal_elements(self):
        ''' Returns all maximal elements of the underlying poset.
        
        Returns
        -------
       list
          all maximal elements of the underlying poset.

        '''
        max_elements = []
        for x in self.elements:
            if self.cover_relations[x] == []:
                max_elements.append(x)
        return max_elements
    
    def minimal_elements(self):
        ''' Returns all minimal elements of the underlying poset.
        
        Returns
        -------
        list
            all minimal elements of the underlying poset.

        '''
        reverse_covers = dag.reverse(self.cover_relations)
        min_elements = []
        for x in self.elements:
            if reverse_covers[x] == []:
                min_elements.append(x)
        return min_elements
   
    def covers(self, x):
        ''' Returns all (upper) covers of element 'x'.

        Parameters
        ----------
        x : element of the underlying poset.

        Returns
        -------
        list
            all upper covers of 'x'.

        '''
        return self.cover_relations[x]
    
      
    def upset(self, x):
        ''' Returns the upset of 'x'.
        
        Parameters
        ----------
        x : element of the underlying poset.

        Returns
        -------
        list
            all elements of the underlying poset which are greater than or 
            equal to 'x'.

        '''
        upset_x = dag.descendants(self.cover_relations, x)
        upset_x.append(x)
        return upset_x
    
    def downset(self, x):
        ''' Returns the downset of 'x'.
            
        Parameters
        ----------
        x : element of the underlying poset.

        Returns
        -------
        list
            all elements of the underlying poset which are less than or equal
            to 'x'.

        '''
        downset_x = dag.ancestors(self.cover_relations, x)
        downset_x.append(x)
        return downset_x
        
##############################################################################
######################### COLORING ###########################################
##############################################################################\
    
    def up_colored(self):
        return list(filter(lambda x : self.coloring[x] in {0,1}, self.elements)) 
    
    def down_colored(self):
        return list(filter(lambda x : self.coloring[x] in {-1,0}, self.elements)) 
              
    
    def color_sum(self):
        ''' Returns the sum colors over all elements.
        Returns
        -------
        int 
            the sum of colors over all elements, where 1 (resp. 0,-1) represents 
            blue (resp. green, red).

        '''
        return sum(self.coloring.values())
    
    def color(self, x):
        ''' Returns the color of 'x'.
        
        Parameters
        ----------
        x : int
            element of underlying poset

        Returns
        -------
        int
            the color of 'x', where 1 (resp. 0,-1) represents blue 
            (resp. green, red).

        '''
        
        return self.coloring[x]
    
##############################################################################
######################### OPTIONS/DISJUNCTIVE SUMMANDS #######################
##############################################################################
           
    def up_options(self): 
        ''' Returns Ups options in the game.
        
        Returns
        -------
        dict
            Ups options in the game keyed by the element whose upset has been
            removed.
            
        '''
        options = {}
        for x in self.up_colored():
            option_elements = list(set(self.elements) - set(self.upset(x)))
            option_coloring = {x: self.color(x) for x in option_elements}
            option_covers = dag.subgraph(self.cover_relations, option_elements) 
            option = UpDown(option_covers, option_coloring, covers = True)
            option.coordinates = {x: self.hasse_coordinates[x] for x in option_elements}
            options[x] = option
        return options

    def down_options(self): 
        ''' Returns Downs options in the game.
        
        Returns
        -------
        dict
            Downs options in the game keyed by the element whose downset has 
            been removed.
            
        '''
        options = {}
        for x in self.down_colored():
            option_elements = list(set(self.elements) - set(self.downset(x)))
            option_coloring = {x: self.color(x) for x in option_elements}
            option_covers = dag.subgraph(self.cover_relations, option_elements) 
            option = UpDown(option_covers, option_coloring, covers = True)
            option.coordinates = {x: self.hasse_coordinates[x] for x in option_elements}
            options[x] = option
        return options
              
##############################################################################    
############################### PLOTTING ######################################
##############################################################################

    
    def gameboard(self, marker = 'o'):
        ''' Plots the game. I.e. the (colored) Hasse diagram of the underlying 
        poset.
        
        Parameters
        ----------
        marker : Type, optional
            The default is 'o'.

        Returns
        -------
        None

        '''
        udp.UpDownPlot(self, marker=marker)      
        plt.show()
        return None
    
##############################################################################    
########################### GAMEPLAY #########################################
##############################################################################
    def play(self, marker='o', agent_1 = None, agent_2 = None):    
        ''' Interactively play the game.
        
        Parameters
        ----------
        marker : TYPE, optional
            DESCRIPTION. The default is 'o'.

        Returns
        -------
        None.
        '''
                    
        #close any open plots.
        plt.close()
        #make sure interactivity is off.
        plt.ioff()
        
        players = ['Up', 'Down']
        if agent_1 == None or agent_2 == None:
            #decide which player is first.
            first = input("Which player will start, 'Up' or 'Down'? ")       
            
            #simple catch if input is incorect.
            while first not in players:
                first = input("Please choose 'Up' or 'Down'! ")
        else:
            first = random.choice(players)
        
        #assign remaining player option to second.
        players.remove(first)
        second = players[0]
        
        #assign color values codes. Up is BG (0,1), down is RG (-1,0)
        if first == 'Up':
            first_colors, second_colors = {0,1}, {-1,0}
        else:
            first_colors, second_colors = {-1,0}, {0,1}
        
        #initialize a player token to keep track of whose turn it is.            
        i = 0
        
        #Initialize the figure.
        #fig_info contains various dicts that point to specific
        # objects in our figure. These are used to remove
        # these objects from the figure as the game progresses.
        board = udp.UpDownPlot(self, marker=marker)                           
        
        #give the game a boring title
        board.figure.suptitle("Upset-Downset")                                 
        
        #shows the initial figure before starting the game.
        plt.pause(0.01)
        
        #The game is a while loop executes as long as the graph has nodes.
        while self.elements:                     
            
            #Players can only choose nodes of their allowed colors.
            #Sort available choices for each player into respective lists.
            first_options = list(filter(lambda x : self.coloring[x] in first_colors, \
                  self.elements))            
            second_options = list(filter(lambda x : self.coloring[x] in second_colors, \
                  self.elements))              
            
            #If either first_options or second_options is empty
            # game will end and the player with the nonempty list will
            # be declared the winner. Note, cannot be inside this while loop
            # and have both first_options and second_options empty.
            if not first_options:
                i=0
                break
                
            if not second_options:
                i=1
                break
                
            #game played when first player is "Up".
            if first == "Up":
                
                #first player's turn
                if i % 2 == 0:
                    
                    if agent_1 == None:
                        #first player chooses a node
                        u = int(input(first + ", choose a Blue or Green node: "))
                        
                        #catch if choice was not valid.
                        while not (u in first_options):
                            print(u, " is not a valid choice.")
                            u = int(input(first + ", choose a Blue or Green node: ")) 
                        
                        #get resulting UpDown after removing upset
                        sub_game = self.up_options()[u]
                        
                        #remove objects from figure and fig_info which are no longer
                        # in play.
                        board.leave_subgraph_fig(sub_game)
                        
                        #make the current game self.
                        self = sub_game
                        
                        #show the current figure.
                        plt.pause(0.01)
                    
                    else:
                        print("The computer is choosing...")
                        #time.sleep(1)
                        
                        u = agent_1(self, first)
                        
                        #get resulting UpDown after removing upset
                        sub_game = self.up_options()[u]
                        
                        #remove objects from figure and fig_info which are no longer
                        # in play.
                        board.leave_subgraph_fig(sub_game)
                        
                        #make the current game self.
                        self = sub_game
                        
                        #show the current figure.
                        plt.pause(0.01)
                #second player's turn
                else:
                    if agent_2 == None:
                        #second player chooses a node
                        u = int(input(second + ", choose a Red or Green node: "))
                        
                        #catch if choice was not valid.
                        while not (u in second_options):
                            print(u, " is not a valid choice.")
                            u = int(input(second + ", choose a valid Red or Green node: "))
                        
                        #get resulting UpDown after removing downset
                        sub_game = self.down_options()[u]
                        
                        #remove objects from figure and fig_info which are no longer
                        # in play.
                        board.leave_subgraph_fig(sub_game)
                        
                        #make the current game self.
                        self = sub_game
                        
                        #show the current figure.
                        plt.pause(0.01)
                    else:
                        print("The computer is choosing...")
                        #time.sleep(1)
                        
                        u = agent_2(self, second)
                        
                        #get resulting UpDown after removing upset
                        sub_game = self.down_options()[u]
                        
                        #remove objects from figure and fig_info which are no longer
                        # in play.
                        board.leave_subgraph_fig(sub_game)
                        
                        #make the current game self.
                        self = sub_game
                        
                        #show the current figure.
                        plt.pause(0.01)
            
            #game played when first player is "Down".
            else:
                
                #first player's turn.
                if i % 2 == 0:
                    if agent_1 == None:
                        #first player chooses a node.
                        u = int(input(first + ", choose a Red or Green node: "))
                        
                        #catch if choice was not valid.
                        while not (u in first_options):
                            print(u, " is not a valid choice.")
                            u = int(input(first + ", choose a Red or Green node: ")) 
                        
                        #get resulting UpDown after removing downset
                        sub_game = self.down_options()[u]
                        
                        #remove objects from figure and fig_info which are no longer
                        # in play.
                        board.leave_subgraph_fig(sub_game)
                        
                        #make the current game self.
                        self = sub_game
                        
                        #show current figure.
                        plt.pause(0.01)
                    else:
                        print("The computer is choosing...")
                        #time.sleep(1)
                        
                        u = agent_1(self, first)
                        
                        #get resulting UpDown after removing upset
                        sub_game = self.down_options()[u]
                        
                        #remove objects from figure and fig_info which are no longer
                        # in play.
                        board.leave_subgraph_fig(sub_game)
                        
                        #make the current game self.
                        self = sub_game
                        
                        #show the current figure.
                        plt.pause(0.01)
                        
                #second player's turn
                else:
                    if agent_2 == None:
                        #second player chooses a node.
                        u = int(input(second + ", choose a Blue or Green node: "))
                        
                        #catch if choice was not valid.
                        while not (u in second_options):
                            print(u, " is not a valid choice.")
                            u = int(input(second + ", choose a valid Blue or Green node: "))
                        
                        #get resulting UpDown after removing upset
                        sub_game = self.up_options()[u]
                        
                        #remove objects from figure and fig_info which are no longer
                        # in play.
                        board.leave_subgraph_fig(sub_game)
                        
                        #make the current game self.
                        self = sub_game
                        
                        #show current figure.
                        plt.pause(0.01)
                    else:
                        print("The computer is choosing...")
                        #time.sleep(1)
                        
                        u = agent_2(self, second)
                        
                        #get resulting UpDown after removing upset
                        sub_game = self.up_options()[u]
                        
                        #remove objects from figure and fig_info which are no longer
                        # in play.
                        board.leave_subgraph_fig(sub_game)
                        
                        #make the current game self.
                        self = sub_game
                        
                        #show the current figure.
                        plt.pause(0.01)
                        
            
            #update player tokern (i.e. change player turn)
            i += 1   
        
        #once outside of while loop, declare the winner based on
        # what value player token was at.
        if i % 2 == 0:
            print(second + " wins!")
            #change figure title to declare the winner.
            board.figure.suptitle(second + " wins!")
        else:
            print(first + " wins!")
            #change figure title to declare the winner.
            board.figure.suptitle(first + " wins!")
        
        #show figure with winner title declaration.
        plt.pause(0.01)
        
##############################################################################    
########### UPDOWNS PARTIALLY ORDERED ABELIAN GROUP STRUCTURE ################
##############################################################################
    
    def outcome(self):
        ''' Returns the outcome of the game. **Due to the huge number of 
        suboptions, for all but the smallest games this algotithm is extremely 
        slow.
        
        Returns
        -------
        str
            'Next', Next player (first player to move) wins.
            'Previous', Previous player (second player to move) wins.
            'Up', Up can force a win. (Playing first or second). 
            'Down', Down can force a win. (Playing first or second). 

        '''
        def option_outcomes(G, elems):
            n = len(G)
            e = dag.number_of_edges(G.cover_relations)
            color_sum = G.color_sum()
            # Possible outcomes
            N, P, L, R = 'Next', 'Previous', 'Up', 'Down'
            # Base cases/heuristics for recursion (either no nodes or no edges):
            # No nodes, second player win.
            if n == 0:
                out = P
            # Nozero # of nodes and all are blue,  Up wins.
            elif color_sum == n:
                out = L
             # nozero # of nodes and all are red,  Down wins.
            elif color_sum == -n:
                out = R
            # nonzero # of nodes, but no edges:
            elif e == 0:
                if color_sum == 0:
                    # No edges, equal # of blue and red nodes, even number of 
                    # nodes, second player win.
                    if n%2 == 0:
                        out = P
                    # No edges, equal # of blue and red nodes, odd number of 
                    # nodes, second player win.
                    else:
                        out = N
                # No edges, more blue nodes than red, Up wins
                elif color_sum > 0:
                    out = L
                # No edges, more red nodes than blue, Down wins
                else:
                    out = R
            # Using memoization, recursively visit G's options.
            else:
                # Outcomes of Ups options in G.
                up_ops_otcms = set()
                # Outcomes of Downs options in G.
                dwn_ops_otcms = set()
                # Determine the outcome of each of Ups options.
                up_ops = G.up_options()
                for x in up_ops:
                    G_L = up_ops[x]
                    sub_elems = frozenset(G_L.elements)
                    # If we've already computed the outcome of this subposition.
                    if sub_elems in sub_outcomes:
                        up_ops_otcms.add(sub_outcomes[sub_elems])
                    # We havent computed the outcome of current subposition.
                    else:
                        up_ops_otcms.add(option_outcomes(G_L, sub_elems))  
                # Same for Downs options
                dwn_ops = G.down_options()
                for x in dwn_ops:
                    G_R = dwn_ops[x]
                    sub_elems = frozenset(G_R.elements)
                    if sub_elems in sub_outcomes:
                        dwn_ops_otcms.add(sub_outcomes[sub_elems])
                    else:
                        dwn_ops_otcms.add(option_outcomes(G_R, sub_elems))
                # Determine outcome of G via the outcomes of the options:
                # First player to move wins.
                if {P, L} & up_ops_otcms and {P,R} & dwn_ops_otcms:
                    out = N
                # Second player to move wins
                elif not {P, L} & up_ops_otcms and not {P, R} & dwn_ops_otcms:
                    out = P
                # Up wins no matter who moves first
                elif {P, L} & up_ops_otcms and not {P,R} & dwn_ops_otcms:
                    out = L
                # Down wins no matter who moves first:
                # not {P, L} & up_ops_otcms and {P,R} & dwn_ops_otcms
                else:
                    out = R
            # Store outcome of G in sub_outcomes
            sub_outcomes[elems] = out
            return out
        # Store elements of G as hashable object for memoization. This works 
        # since there is a unique subposition for each subset of elements.
        elems = frozenset(self.elements)
        # Dict to store outcomes of subpositions keyed by (hashable) elements of 
        # subposition.
        sub_outcomes = {}
        # Now recursively determine outcome.
        out = option_outcomes(self, elems)
        return out
    
    def __neg__(self):                
        '''Returns the negative of the game.
    
        Returns
        -------
        UpDown
            the upset-downset game on the dual poset and opposite coloring.

        '''
        dual_covers = dag.reverse(self.cover_relations)
        reverse_coloring = {x: -self.coloring[x] for x in self.elements}
        dual_height =self.height()
        dual_hasse_coords = {x:(self.hasse_coordinates[x][0], dual_height - \
                                self.hasse_coordinates[x][1]) for x in self.elements}
        negative = UpDown(dual_covers, reverse_coloring, covers = True)
        negative.hasse_coordinates = dual_hasse_coords
        return negative

    def __add__(self, other):              
        '''Returns the (disjunctive) sum of games. **Relabels elements in 'other'
        to consecutive nonnegative integers starting from len('self').
        
        Parameters
        ----------
        other : UpDown

        Returns
        -------
        UpDown
            The upset-downset game on the disjoint union of posets with 
            unchanged colorings.
            
        Note: the sum retains all elements ('other' being relabelled), relations 
            and coloring from both 'self' and 'other' with the added relations that all 
            elements of 'self' and 'other' are incomparable.

        '''
        # initialze  cover relations in sum and update with selfs relations
        sum_covers = {}
        sum_covers.update(self.cover_relations)
        # initialze coloring in sum and update with selfs relations
        sum_coloring = {}
        sum_coloring.update(self.coloring)
        # initialze dict for hasse coordinates in sum and update with selfs 
        # hasse coordinates
        sum_hasse_coords = {}
        sum_hasse_coords.update(self.hasse_coordinates)
        # Relabelling for other.
        n = len(self)
        other_relabelling = dag.integer_relabel(other.cover_relations, n)
        # update cover relations in sum with others relations
        sum_covers.update(other_relabelling['relabelled graph'])
        # update coloring/hasse coordinates of sum with others colors/hasse coordinates
        other_relabel_map = other_relabelling['relabel map']
        for x in other_relabel_map:
            y = other_relabel_map[x]
            sum_coloring[y] = other.coloring[x]
            sum_hasse_coords[y] = (other.hasse_coordinates[x][0] + n, \
                                   other.hasse_coordinates[x][1])
        # initailze the sum and update hasse_coordinates
        SUM = UpDown(sum_covers, sum_coloring, covers = True)
        SUM.hasse_coordinaes = sum_hasse_coords
        return SUM
    
    def __sub__(self, other):
        ''' Returns the difference of games.
    
        Parameters
        ----------
        other : UpDown

        Returns
        -------
        UpDown
            the upset-downset game on the disjoint union of the poset underliying
            'self' with unchanged coloring and the dual of the poset underlying 
            'other' with the opposite coloring.

        '''
        return self + (-other)
    
    def __eq__(self, other):
        ''' Returns wether the games are equal. ** Depends on the outcome method.

        Parameters
        ----------
        other : UpDown

        Returns
        -------
        bool
            True if the games 'self' and 'other' are equal (their differecne is 
            a second player win) and False otherwise.

        '''
        return (self - other).outcome() == 'Previous'
    
    def __or__(self, other):
        ''' Returns wether games are incomparable (fuzzy). ** Depends on the 
        outcome method.

        Parameters
        
        ----------
        other : UpDown object

        Returns
        -------
        bool
            True if the games 'self' and 'other' are fuzzy (their difference is 
            a first player win) and False otherwise.
        '''
        return (self - other).outcome == 'Next'

    def __gt__(self, other):
        ''' Returns wether games are comparable in specified order. ** Depends 
        on the outcome method.

        Parameters
        ----------
        other : UpDown object

        Returns
        -------
        bool
            True if 'self' is greater than 'other' (better for Up: their 
            differnce is a win for Up)  and False otherwise.

        '''
        return (self - other).outcome() == 'Up'
    
    def __lt__(self, other):
        ''' Returns wether games are comparable in specified order. ** Depends 
        on the outcome method.

        Parameters
        ----------
        other : UpDown object

        Returns
        -------
        bool
            True if 'self' is less than 'other' (better for Down: their 
            differnce is a win for Down) and False otherwise.

        '''
        return (self - other).outcome() == 'Down'
    
    def disjunctive_summands(self):             #FINISH FOR GODS SAKE!!!!!!!!!!!!!!!!
        ''' TO BE WRITTEN... 

        Returns
        -------
        None.

        '''
        return None

###############################################################################
######################### NEW GAMES FROM OLD ##################################
##############################################################################
    

    def __truediv__(self, other):          
        ''' Returns the ordinal sum of games 'G' and 'H'. (This is not a 
        commutative operation). **Relabels all elements of 'self' with consecutive 
        nonnegative integers starting from len('other')
        
        Parameters
        ----------      
        other: UpDown object

        Returns
        -------
        UpDown
            The ordinal sum retains all elements ('self' relabelled), relations and
            coloring from both 'self' and 'other' with the added relations that every
            element of 'self' is greater than every element of 'other'

        '''
        # initialze cover relations in ordinal sum and updates with
        # relations in other (before adding new relations between other and self).
        ordinal_covers = {}
        # Do not want to mutate others relations later!
        for x in other.elements:
            ordinal_covers[x] = []
            for y in other.cover_relations[x]:
                ordinal_covers[x].append(y)
        # initialze coloring in ordinal sum and updates with other coloring.
        ordinal_coloring = {}
        ordinal_coloring.update(other.coloring)
        # initialze hasse coordinates in ordinal sum and update with others
        # coordinates. 
        ordinal_hasse_coords = {}
        ordinal_hasse_coords.update(other.hasse_coordinates)
        # Relabelling for self.
        n = len(other)
        self_relabelling = dag.integer_relabel(self.cover_relations, n)
        self_relabel_map = self_relabelling['relabel map']
        # update cover relations in ordinal sum with selfs relations
        ordinal_covers.update(self_relabelling['relabelled graph'])
        # update cover relations in ordinal sum with new relations between self and other. 
        other_maximal = other.maximal_elements()
        self_minimal = [self_relabel_map[x] for x in self.minimal_elements()]
        for x in other_maximal:
            ordinal_covers[x].extend(self_minimal)
        # update coloring of ordinal sum with selfs coloring
        self_colors = {self_relabel_map[x]: self.coloring[x] for x in self.elements}
        ordinal_coloring.update(self_colors)
        # update hasse coordinatesin ordinal sum with coordinates for self
        other_x_coords = [other.hasse_coordinates[x][0] for x in other.elements]
        MAX = max(other_x_coords)
        other_height = other.height()
        self_hasse_coords = \
            {self_relabel_map[x]: (self_relabel_map[x] - MAX, \
                                   self.hasse_coordinates[x][1] + other_height+1) for x in self.elements}
        ordinal_hasse_coords.update(self_hasse_coords)
        # initialize the ordinal sum and update the hasse coordinates
        ordinal_sum = UpDown(ordinal_covers, ordinal_coloring, covers = True)
        ordinal_sum.hasse_coordinates = ordinal_hasse_coords
        return ordinal_sum
