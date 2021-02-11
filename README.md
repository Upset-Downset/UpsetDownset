# UpDown

Authors: Charles Petersen and Jamison Barsotti

Upset-downset is a two player game created by Tim Hsu in which the players alternate turns 
deleting nodes from graphs whose vertices are colored blue, green or red. The Up player moves 
by deleting a blue or green node, together with any nodes connected to it by a path moving 
strictly upward. (All nodes above the chosen node are removed regardless of their color.) Similarly, 
the Down player moves by deleting a red or green node, together with any nodes connected to it 
by a path moving strictly downward. (All nodes below the chosen node are removed regardless of 
their color.)  Eventually one of the players will find they cannot move because there are no longer
any nodes of their color. Whoever is first to find themselves in this predicament loses.  

Goal: develop and train an machine learning system that learns to play upset-downset well, based on the principals of reinforcement learning.
    
Contents: To be added...

TO DO LIST: (2/5/21)

    - (BASICALLY DONE) Sufficiently abstract the UpDownNet: input block, res blocks and output block? 
    - (DONE) Implement uniform poset generation into randomUpDown. 
    - Setup code to save and plot training data: training loss, policy loss, value loss,
    evaluation score...
    - (DONE) Should we store all teaing data (self_play_data, eval_data, model_data) in a folder
    'train_data' with a text file containing the paramaters for the training? 
    - (DONE EXCEPT IN play() method) switch to f-strings everywhere. In particular use f-string in the play() method.
    - Compactify the play() method?
    - Finish Upset-Downset intro notebook.
    - Finish game generation notebook.
    - Start/Finish trainig method notebook.
    - Outcomes notebook?
    - Update this read me...
        * overview of project
        * where project stands/ what's next.
        * brief description of each module

TO DO LIST: (2/11/21)

    - (BASICALLY DONE, need to implement new model across all necessary modules and have it interatc with global variables) 
    Sufficiently abstract the UpDownNet: input block, res blocks and output block? 
    - Setup code to save and plot training data: training loss, policy loss, value loss,
    evaluation score...
    - (DONE EXCEPT IN play() method) Switch to f-strings everywhere. In particular use f-strings in the play() method.
    - Compactify the play() method?
    - Finish Upset-Downset intro notebook.
    - Finish game generation notebook.
    - Start/Finish trainig method notebook.
    - Outcomes notebook?
    - Update this read me...
        * overview of project
        * where project stands/ what's next.
        * brief description of each module
    - Finish abstraction of function gameState.py to a class 'GameState' and implement changes across all necessary modules.
    - Should we be saving the new replay buffer after each training (just overwrite each time) so that if stop trainng we can always come back the
    last replay buffer?
    - Need to fix nimUpDown/completeBipartiteUpDown so that their options are also of the class nimGame/completeBipartiteGame. I have an idea how to 
    to do this: for each instance of nimGame/completeBipartiteGame have an attrribute which is a dictinary: heap/graph in the instance is a 
    key and the corresponding value is the list of all nodes contained in the corresponding heap/graph. when instantiating an option (ie. the up_play()
    or down_play() methods) this info can be passed to the function which 
    generates the relations so as to keep track of node labels in the option... me thinks this should work, even though it sound ugly and i have'nt 
    gone through the deatils.
    
    ****** There is a new branch called charlespetersen-patch-1:
    ****** This branch contains:

        - changed the name of the module housing the functions for the Markov process to randomDag.py
        - updated randomUpDown.py to use the Markov process to produce random games. 
        - updated gameState.py to include a generator which produces random games via a continuing Markov process.The generator has not been
        implemented in self-play or evaluation yet.
        - in utils.py I've added a function to find the latest dag(though transitively reduced) prided in the Markov chain taking place in 
        whatever proceed of multiprocess
        - in trainPipeine.py and train.py I've implemented a replay buffer for training. 
        - in selfPlay.py, evalPlay.py, utils.py, train.py and trainPipeline.py I implelented f-strings everywhere.
        - began working on an abstraction of the functions contained in  gameState.py to a  class. This module is named `gameState_abstract.py`.
        - sufficiently abstracted the model to have an easily changeable architecture. I have not implemented the new version of the model in any 
        files where it is needed. The module containing the abstracted model is in `model_abstract.py`.
        - updated utils.py and all necessary modules so that all directories/files produced during training will now be kept in a single 
        directory called 'train_data'.

