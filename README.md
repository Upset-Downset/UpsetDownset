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

    - Sufficiently abstract the UpDownNet: input block, res blocks and output block?
    - Implement uniform poset generation into randomUpDown
    - Setup code to save and plot training data: training loss, policy loss, value loss,
    evaluation score...
    - Should we store all teaing data (self_play_data, eval_data, model_data) in a folder
    'train_data' with a text file containing the paramaters for the training?
    - switch to f-strings everywhere. In particular use f-string in the play() method.
    - Compactify the play() method?
    - Finish Upset-Downset intro notebook.
    - Finish game generation notebook.
    - Start/Finish trainig method notebook.
    - Outcomes notebook?
    - Update this read me...
        * overview of project
        * where project stands/ what's next.
        * brief description of each module


