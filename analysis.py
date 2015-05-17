# utilities for analyzing the chess learning system.
import numpy as np
from matplotlib import pyplot as plt
from sunfish.xboard import parseFEN
from execution import PIECE_DICT

# function definitions
# parse 
def parse(fen):
    rows = parseFEN(fen).board.split('\n')[1:9]
    stripped = [r.strip() for r in rows]
    pieces = [[PIECE_DICT[p] for p in list(r)] for r in stripped]
    return np.asarray(pieces)

def render_state(fen,c=1):
    plt.imshow(parse(fen).transpose(2,0,1)[c,:,:], interpolation='nearest')

 
