#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 22:41:10 2020

@author: Charlie & Jamison
"""

import matplotlib.pyplot as plt
import poset as pst
import HarryPlotter as hp

class UpDown(object):
    ''' Abstract class for construction of an upset downset game from 
    red-green-blue colored poset. Not to be accessed directly, but through 
    the subclass heirarchy via creation of an UpDown sublass.
    '''
    def __init__(self, poset):
        '''
        Parameters
        ----------
        poset : Poset object ** See Poset
 
        Returns
        -------
        None

        '''
        self._poset = poset
            
    def poset(self):
        ''' Returns the underlying poset.

        Returns
        -------
        Poset object    

        '''
        return self._poset
             
    def upOptions(self):
        ''' Returns Ups options in the game.
        
        Returns
        -------
        dict
            Ups options in the game keyed by the corresponding element of the 
            underlying poset. 
            
        '''
        up_options = {}
        nodes = set(self._poset.elements())
        for x in nodes:
            if self._poset.color(x) in {0,1}:
                upset = set(self._poset.upset(x))
                option_nodes = nodes-upset
                option_nodes = list(option_nodes)
                option_poset = self._poset.subposet(option_nodes) 
                up_options[x] = UpDown(option_poset)
        return up_options

    def downOptions(self):
        ''' Returns Downs options in the game.
        
        Returns
        -------
        dict
            Downs options in the game keyed by the corresponding element of the 
            underlying poset.
            
        '''
        down_options = {}
        nodes = set(self._poset.elements())
        for x in nodes:
            if self._poset.color(x) in {-1,0}:
                downset = set(self._poset.downset(x))
                option_nodes = nodes-downset
                option_nodes = list(option_nodes)
                option_poset = self._poset.subposet(option_nodes) 
                down_options[x] = UpDown(option_poset)
        return down_options
    
    def gameboard(self, marker = 'o'):
        ''' Plots the game. I.e. the Hasse diagram of the underlying poset.
        
        Parameters
        ----------
        marker : Type, optional
            DESCRIPTION. The default is 'o'.

        Returns
        -------
        None

        '''
        # HARRY PLOTTER.... HASSE METHOD FROM POSET CLASS...?
        return None
    
    
    def play(self, marker='o'):  #NEED TO UPDATE AND ADD COMMENTS
        ''' Interactively play the game.
        
        Parameters
        ----------
        marker : TYPE, optional
            DESCRIPTION. The default is 'o'.

        Returns
        -------
        None.
        
        # I'd like to add some functionality to the play function at some point
        #    * Be able to exit the loop at the current position while
        #      simultaneously initializing an upDown object corresponding to the 
        #      current position. (For investigating outcomes.)

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
        fig_info = hp.harry_plotter(self, marker=marker)
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
            print(second + " wins!")
            fig_info[0].suptitle(second + " wins!")
        else:
            print(first + " wins!")
            fig_info[0].suptitle(first + " wins!")
        plt.pause(0.01)
    
    
    def outcome(self):
        ''' Returns the outcome of the game. **Due to the huge number of 
        suboptions, for all but the smallest games this algotithm is extremely 
        slow.
        
        Returns
        -------
        str
            'Next', Next player (first playr to move) wins.
            'Previous', Previous player (second player to move) wins.
            'Up', Up can force a win playing first or second. 
            'Down', Down corce a win playing first or second. 

        '''
        n = len(self._poset)
        e = self._poset.coverSum() 
        N, P, L, R = 'Next', 'Previous', 'Up', 'Down'
        # Base cases for recursion
        colors_sum = self._poset.colorSum() 
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
            # I can make this slightly faster
            up_ops_otcms = {G.outcome() for G in self.upOptions().values()}
            dwn_ops_otcms = {G.outcome() for G in self.downOptions().values()}
            # First player to move wins. 
            if (P in up_ops_otcms or L in up_ops_otcms) and \
                (P in dwn_ops_otcms or R in dwn_ops_otcms):
                out = N
            # Up wins no matter who moves first
            elif (P in up_ops_otcms or L in up_ops_otcms) and \
                (P not in dwn_ops_otcms and R not in dwn_ops_otcms):
                out = L
            # Down wins no matter who moves first
            elif (P in dwn_ops_otcms or R in dwn_ops_otcms) and \
                (P not in up_ops_otcms and L not in up_ops_otcms):
                out = R
            # Second player to move wins
            elif (P not in up_ops_otcms and L not in up_ops_otcms) and \
                (P not in dwn_ops_otcms and R not in dwn_ops_otcms):
                out = P
        return out
    
    def __neg__(self):
        '''Returns the negative of the game.
    
        Returns
        -------
        UpDown object
            the upset-downset game on the dual poset and opposite coloring.

        '''
        colored_dual = self._poset.coloredDual()
        return UpDown(colored_dual)

    def __add__(self, other):
        '''Returns the sum of games. **Relabels nodes to consecutive 
        nonnegatove integers starting from 0.
        
        Parameters
        ----------
        other : UpDown object

        Returns
        -------
        UpDown object
            The upset-downset game on the disjoint union of posets with 
            unchanged colorings.

        '''
        disjoint_union = self._poset + other.poset
        return UpDown(disjoint_union)
    
    def __sub__(self, other):
        ''' Returns the difference of games.
    
        Parameters
        ----------
        other : UpDown object

        Returns
        -------
        UpDown object
            the upset-downset game on the disjoint union of the poset underliying
            'self' with unchanged coloring and the dual of the poset underlying 
            'other' with the opposite coloring.

        '''
        return self + (- other)
    
    def __eq__(self, other):
        ''' Returns wether the games are equal. ** Depends on the outcome method.

        Parameters
        ----------
        other : UpDown object

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
