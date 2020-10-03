# UpDown

Authors: Charles Petersen and Jamison Barsotti

The game of upset-downset:

Let P be a finite poset with coloring c:P -> {-1,0,1} where 1 (resp. 0,-1) 
represent blue (resp. green, red). The upset-downset game on P is the 
partizan combinatorial game with the following possible moves: For any element x in P colored blue or green, Left (Up) may remove the upset >x of x (>x := {y in P : y>=x}) leaving the colored poset P - >x, and for x in P colored red or green, Right (Down) may remove the downset <x of x (<x := {y in P : y<=x}) leaving the colored poset P - <x. The first player unable to move loses. 

Upset Downset is usually played on the Hasse diagram of P. The available 
moves for Up and Down can then be described as: Up may choose to remove any blue or green colored vertex on the Hasse diagram of P along with all vertices connected to it by a path moving strictly upward, and Down may choose to remove any red or green colored vertex on the Hasse diagram of P along with all vertices connected to it by a path moving strictly downward.
The first player who cannot remove any vertices loses. 

Goals:
    * Create an interface that allows for construction and experimentation of 
    upset-downset games on various posets.
    * Develop and train an AI system that learns to play upset-downset well, based 
    on principals of reinforment learning.
    
Contents: To be added...


