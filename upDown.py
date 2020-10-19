#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 22:41:10 2020

@author: Charlie & Jamison
"""

import dagUtility as dag
import upDownPlot as uDP
import matplotlib.pyplot as plt

class UpDown(object):
    ''' Abstract class for construction of an upset-downset game from a 
    blue-green-red partially ordered set (poset). Not to be accessed directly, 
    but through the subclass heirarchy via creation of an UpDown sublass on a 
    specific  poset or family of posets.
    '''
    def __init__(self, relations, coloring, covers = False):
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
                - Reflexivity is assumed. Do not provide the reflexive relations.
                  (No loops in the corresponding dag.)
                - Anti-symmetry is checked via acylicity of the corresponding dag.
                - Transitivity is forced via reachability in the corresponding dag.
                  (In particular, the poset returned is the transitive closure of the 
                  provided 'relations'.)
            We identify posets with dags as usual:
                    nodes of dag <----> elements of poset 
                    reachability in dag <----> relation in poset 
                    (b reahcbale from a in dag <----> a<b in poset) 
        coloring: dict
            coloring map on elements in underlying poset. color keyed by element
            where 1 (resp. 0, -1) represents blue (resp. green, red).
        covers : bool, optional
            set to 'True' if the given 'relations' are the (upper) covering 
            relations on the underlying poset. (In this case, anti-symmetry will 
            not be checked.) The default is False.
        '''
        if not covers:
            assert dag.is_acyclic(relations), 'The given relations are not anti-symmetric.'
            self._cover_relations = dag.transitive_reduction(relations)
        else:
            self._cover_relations = relations
        self._elements = list(self._cover_relations)
        self._coloring_map = coloring
        
##############################################################################
######################### POSET ##############################################
##############################################################################
        
    def cover_relations(self):
        ''' Returns the (upper) cover relations of the underlying poset.

        Returns
        -------
        dict
            list of (upper) covers keyed by element.
            
        '''
        return self._cover_relations
        
    def elements(self):
        ''' Returns all elements of the underlying poset.
    
        Returns
        -------
        list
            all elements of the underlying poset

        '''
        return self._elements
    
    def __len__(self):
        ''' Returns the cardinality of the underlying poset. 

        Returns
        -------
        int
            number of elements in the underlying poset.

        '''
        return len(self._elements)
    
    def levels(self):
        ''' Returns the level of each element in the Hasse diagram of the 
        underlying poset.

        Returns
        -------
        dict
            keyed by elements of the underlying poset, with corresponding value 
            being the length of a longest path in the Hasse diagram which 
            terminates in the key.

        '''
        return dag.longest_path_lengths(self._cover_relations, direction = 'incoming')
        
    def height(self):
        ''' Returns the height of the underlying poset.
        
        Returns
        -------
        int
            the maximum level in the underlying poset. ** See levels().

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
        for x in self._elements:
            if self._cover_relations[x] == []:
                max_elements.append(x)
        return max_elements
    
    def minimal_elements(self):
        ''' Returns all minimal elements of the underlying poset.
        
        Returns
        -------
        list
            all minimal elements of the underlying poset.

        '''
        reverse_covers = dag.reverse(self._cover_relations)
        min_elements = []
        for x in self._elements:
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
        return self._cover_relations[x]
    
      
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
        upset_x = dag.descendants(self._cover_relations, x)
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
        downset_x = dag.ancestors(self._cover_relations, x)
        downset_x.append(x)
        return downset_x
        
##############################################################################
######################### COLORING ###########################################
##############################################################################
        
    def coloring_map(self):
        ''' Returns the coloring map on the game.

        Returns
        -------
        dict
            color keyed by element of the underlying poset where 1 (resp. 0,-1)
            represents blue (resp. green, red).

        '''
        return self._coloring_map
    
    def color_sum(self):
        ''' Returns the sum over all colors in the game.
        Returns
        -------
        int 
            the sum over all colors in the game where 1 (resp. 0,-1) represents 
            blue (resp. green, red).

        '''
        return sum(self._coloring_map.values())
    
    def color(self, x):
        ''' Returns the color of 'x'.
        
        Parameters
        ----------
        x : element of underlying poset

        Returns
        -------
        int
            the color of 'x', where 1 (resp. 0,-1) represents blue 
            (resp. green, red).

        '''
        
        return self._coloring_map[x]
    
##############################################################################
######################### SUBPOSITIONS/OPTIONS ETC ###########################
##############################################################################
   
    def subpositions(self, other_arguments):
       ''' TO BE WRITTEN... H is a subposition of G if there is a sequence of 
       moves (not necessarily alternating) leading from G to H. I.e., subpositions
       are subposets...
       
       Returns
       -------
       None.

       '''
       return None
          
    def up_options(self): 
        ''' Returns Ups options in the game.
        
        Returns
        -------
        dict
            Ups options in the game keyed by the element whose upset has been
            removed.
            
        '''
        options = {}
        for x in self._elements:
            if self.color(x) in {0,1}:
                option_elements = list(set(self._elements) - set(self.upset(x)))
                option_coloring = {y: self.color(y)  for y in option_elements}
                option_covers = dag.subgraph(self._cover_relations, option_elements) 
                options[x] = UpDown(option_covers, option_coloring, covers = True)
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
        for x in self._elements:
            if self.color(x) in {-1,0}:
                option_elements = list(set(self._elements) - set(self.downset(x)))
                option_coloring = {y: self.color(y) for y in option_elements}
                option_covers = dag.subgraph(self._cover_relations, option_elements) 
                options[x] = UpDown(option_covers, option_coloring, covers = True)
        return options
    
    def summands(self):
        ''' TO BE WRITTEN... 

        Returns
        -------
        None.

        '''
        return None
                
##############################################################################    
############################### PLOTTING ######################################
##############################################################################
    
    def gameboard(self, marker = 'o'):
        ''' Plots the game. I.e. the (colored) Hasse diagram of the underlying 
        poset.
        
        Parameters
        ----------
        marker : Type, optional
            DESCRIPTION. The default is 'o'.

        Returns
        -------
        None

        '''
        uDP.up_down_plot(self, marker=marker)
        plt.show()
        return None
    
##############################################################################    
########################### GAMEPLAY #########################################
##############################################################################
     
    def play(self, marker='o'):
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
        
        #decide which player is first.
        players = ['Up', 'Down']
        first = input("Which player will start, 'Up' or 'Down'? ")       
        
        #simple catch if input is incorect.
        while first not in players:
            first = input("Please choose 'Up' or 'Down'! ")
        
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
        fig_info = uDP.up_down_plot(self, marker=marker)
        
        #give the game a boring title
        fig_info[0].suptitle("Current Game")
        
        #shows the initial figure before starting the game.
        plt.pause(0.01)
        
        #The game is a while loop executes as long as the graph has nodes.
        while self._cover_relations.keys():           
            
            #Players can only choose nodes of their allowed colors.
            #Sort available choices for each player into respective lists.
            first_options = list(filter(lambda x : self._coloring_map[x] in first_colors, \
                  list(self._cover_relations.keys())))
            second_options = list(filter(lambda x : self._coloring_map[x] in second_colors, \
                  list(self._cover_relations.keys())))
            
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
                    uDP.leave_subgraph_fig(sub_game, fig_info)
                    
                    #make the current game self.
                    self = sub_game
                    
                    #show the current figure.
                    plt.pause(0.01)
                    
                #second player's turn
                else:
                    
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
                    uDP.leave_subgraph_fig(sub_game, fig_info)
                    
                    #make the current game self.
                    self = sub_game
                    
                    #show the current figure.
                    plt.pause(0.01)
            
            #game played when first player is "Down".
            else:
                
                #first player's turn.
                if i % 2 == 0:
                    
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
                    uDP.leave_subgraph_fig(sub_game, fig_info)
                    
                    #make the current game self.
                    self = sub_game
                    
                    #show current figure.
                    plt.pause(0.01)
                
                #second player's turn
                else:
                    
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
                    uDP.leave_subgraph_fig(sub_game, fig_info)
                    
                    #make the current game self.
                    self = sub_game
                    
                    #show current figure.
                    plt.pause(0.01)
            
            #update player tokern (i.e. change player turn)
            i += 1   
        
        #once outside of while loop, declare the winner based on
        # what value player token was at.
        if i % 2 == 0:
            print(second + " wins!")
            #change figure title to declare the winner.
            fig_info[0].suptitle(second + " wins!")
        else:
            print(first + " wins!")
            #change figure title to declare the winner.
            fig_info[0].suptitle(first + " wins!")
        
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
            'Next', Next player (first playr to move) wins.
            'Previous', Previous player (second player to move) wins.
            'Up', Up can force a win. (Playing first or second). 
            'Down', Down corce a win. (Playing first or second). 

        '''
        n = len(self)
        e = dag.number_of_edges(self._cover_relations)
        N, P, L, R = 'Next', 'Previous', 'Up', 'Down'
        # Base cases for recursion
        colors_sum = self.color_sum()
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
        # Recursively determine the ouctome from the games options...
        else:     
            # Set of the outcomes of Ups options
            up_ops_otcms = set()
            # Set of the outcomes of Downs options
            dwn_ops_otcms = set()
            # Recursively determine the outcome of each of Ups options.
            ups_ops = self.up_options()
            for x in ups_ops:
                option = ups_ops[x]
                up_ops_otcms.add(option.outcome())
                # If both a second player win and a win for Up appear as options 
                # outcomes we can stop looking.
                if {P,L} in up_ops_otcms:
                    break
            # Same for Downs options
            dwn_ops = self.down_options()
            for x in dwn_ops:
                option = dwn_ops[x]
                dwn_ops_otcms.add(option.outcome())
                if {P,R} in dwn_ops_otcms:
                    break
            # Determine outcome via the outcomes of the options:
            # First player to move wins. 
            if {P, L} & up_ops_otcms and {P,R} & dwn_ops_otcms:
                out = N
            # Second player to move wins
            elif {P, L} not in up_ops_otcms and {P,R} not in dwn_ops_otcms:
                out = P
            # Up wins no matter who moves first
            elif {P, L} & up_ops_otcms and {P,R} not in dwn_ops_otcms:
                out = L
            # Down wins no matter who moves first:
            # {P, L} not in up_ops_otcms and {P,R} & dwn_ops_otcms
            else:
                out = R
        return out
    
    def __neg__(self):
        '''Returns the negative of the game.
    
        Returns
        -------
        UpDown
            the upset-downset game on the dual poset and opposite coloring.

        '''
        dual_covers = dag.reverse(self._cover_relations)
        reverse_coloring = {x: -self._coloring_map[x] for x in self._elements}
        return UpDown(dual_covers, reverse_coloring, covers = True)

    def __add__(self, other):
        '''Returns the (disjunctive) sum of games. **Relabels elements to consecutive 
        nonnegative integers starting from 0.
        
        Parameters
        ----------
        other : UpDown

        Returns
        -------
        UpDown
            The upset-downset game on the disjoint union of posets with 
            unchanged colorings.
            
        Note: the sum retains all elements (relabelled), relations and coloring 
            from both 'self' and 'other' with the added relations that all 
            elements of 'self' and 'other' are incomparable.

        '''
        sum_covers = {}
        sum_coloring = {}
        # Relabelling for self.
        self_relabel = {}
        self_new_label = 0
        for x in self._elements:
            self_relabel[x] = self_new_label
            self_new_label += 1
        # Relabelling for other.
        other_relabel = {}
        other_new_label = self_new_label
        for x in other.elements():
            other_relabel[x] = other_new_label
            other_new_label += 1
        # Construct the cover relations for disjoint union
        for x in self._elements:
            sum_covers[self_relabel[x]] = []
            sum_coloring[self_relabel[x]] = self._coloring_map[x]
            for y in self._cover_relations[x]:
                sum_covers[self_relabel[x]].append(self_relabel[y])
        for x in other.elements():
            sum_covers[other_relabel[x]] = []
            sum_coloring[other_relabel[x]] = other.color(x)
            for y in other.covers(x):
                sum_covers[other_relabel[x]].append(other_relabel[y]) 
        return UpDown(sum_covers, sum_coloring, covers = True)
    
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
        return self + (- other)
    
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
            differnce is a win for Down)  and False otherwise.

        '''
        return (self - other).outcome() == 'Down'

###############################################################################
######################### NEW GAMES FROM OLD ##################################
##############################################################################
    
    @staticmethod
    def ordinal_sum(G, H):
        ''' Returns the ordinal sum of games 'G' and 'H'. (This is not a 
        commutative operation). **Relabels all elemnts with consecutive 
        nonnegative integers starting from 0.
        
        Parameters
        ----------
        G: UpDown
        
        H: UpDOwn

        Returns
        -------
        UpDown
            The ordianl sum retains all elements (relabelled), relations and
            coloring from both 'G' and 'H' with the added relations that every
            element of 'self' is greater than every element of 'other'

        '''
        ordinal_covers = {}
        ordinal_coloring = {}
        # Relabelling for other.
        H_relabel = {}
        H_new_label = 0
        for x in H.elements():
            H_relabel[x] = H_new_label
            H_new_label += 1
        # Relabelling for self.
        G_relabel = {}
        G_new_label = H_new_label
        for x in G.elements():
            G_relabel[x] = G_new_label
            G_new_label += 1
        # Construct the cover relations for ordianl sum. 
        H_maximal = H.maximal_elements()
        G_minimal  = G.minimal_elements()
        for x in G.elements():
            ordinal_covers[G_relabel[x]] = []
            ordinal_coloring[G_relabel[x]] = G.color(x)
            for y in G.covers(x):
                ordinal_covers[G_relabel[x]].append(G_relabel[y])
        for x in H.elements():
            ordinal_covers[H_relabel[x]] = []
            ordinal_coloring[H_relabel[x]] = H.color(x)
            for y in H.covers(x):
                ordinal_covers[H_relabel[x]].append(H_relabel[y]) 
            if x in H_maximal:
                for z in G_minimal:
                    ordinal_covers[H_relabel[x]].append(G_relabel[z])
        return UpDown(ordinal_covers, ordinal_coloring, covers = True)
    
    @staticmethod
    def fuse(G, H, x, y):
        '''  TO BE WRITTEN.
        
        Parameters
        ----------
        other : Poset
        x : element of poset underlying 'G'.
        y : element of poset underlying 'H'.

        Returns
        -------
        Poset

        '''
        return None
    
