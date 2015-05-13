# utilities for analyzing the chess learning system.
from chess import ChessGame
import numpy as np
from matplotlib import pyplot as plt
from sunfish.xboard import parseFEN

# function definitions
# parse 
def parse(fen):
    rows = parseFEN(fen).board.split('\n')[1:9]
    stripped = [r.strip() for r in rows]
    pieces = [[PIECE_DICT[p] for p in list(r)] for r in stripped]
    return np.asarray(pieces)

def render_state(fen,c=1):
    plt.imshow(parse(fen).transpose(2,0,1)[c,:,:], interpolation='nearest')

# MAIN
g = ChessGame()
N = 20

# run 20 random games.
for i in range(N):
    j = 0
    while not g.is_terminal():
        j += 1
        a = np.random.randint(g.n_actions)
        r = g.take_action(a)
        if j % 50 == 0:
            print "game: %d, iteration: %d, action: %d, reward: %d, status: %d" %(
                i, j, a, r, g._STATUS
            )
    print "game end: %d" %g._STATUS
    g.reset()

 
