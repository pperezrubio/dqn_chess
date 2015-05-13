from Chessnut import Game
from Chessnut.game import InvalidMove
from sunfish.xboard import parseFEN, mrender
from sunfish.sunfish import search,render
from pprint import pprint
import numpy as np
import string
import pylab

# GLOBAL

# PIECE_DICT maps from FEN notation for a piece to 
# my representation of a piece. For instance, a black pawn 'p' 
# is represented by the tuple (0,1), where the first component indicates
# that it is a pawn and the second component indicates that it is black.
# The chess board will be represented as an 8x8 matrix of these two-tuples. 
PIECE_DICT = {'p':(0,1),'r':(1,1),'n':(2,1),'b':(3,1),'q':(4,1),'k':(5,1),
              'P':(0,0),'R':(1,0),'N':(2,0),'B':(3,0),'Q':(4,0),'K':(5,0),
              '.':(-1,-1)}

# These dictionaries map from matrix components to FEN notation and back.
# e.g. to encode the fen move notation 'a2a3' as  matrix locations
XIDX_DICT = {i:c for i,c in zip(range(8),string.lowercase[:8])}
FENX_DICT = {c:i for (i,c) in XIDX_DICT.items()}
YIDX_DICT = {i:j for i,j in zip(range(8),map(str,range(8,0,-1)))}
FENY_DICT = {j:i for (i,j) in YIDX_DICT.items()}


# FEN notation for the initial state of a game board.
INIT_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

# Assignemnt of id to each piece
# - enables us to keep track of particular pieces on the board.
# - enables us to encode moves as offsets for particular pieces.
plocs = [('p'+str(i), (6,i)) for i in range(8)]
nlocs = [('n0', (7,1)),('n1', (7,6))]
blocs = [('b0', (7,2)), ('b1', (7,5))] 
rlocs = [('r0', (7,0)),('r1',(7,7))] 
qklocs = [('q', (7,3)), ('k', (7,4))]
locs = plocs + nlocs + blocs + rlocs + qklocs
PIECE_LOC = {}
LOC_PIECE = {}
for t in locs:
    p,l = t
    PIECE_LOC[p] = l
    LOC_PIECE[l] = p

####################
##  Offset encoding
###################
# Include all offsets that are possible for a piece type in some scenario.
# Ignore castling, promotion, en passant.
# encode these offsets as changes in matrix coordinates. 
poffs = [(-1,0),(-2,0),(-1,-1),(-1,1)] # poffs = pawn offsets
roffs = [x for s in [[(-i,0),(i,0),(0,-i),(0,i)] for i in range(1,8)] for x in s] #roffs = rook offsets
noffs = [(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)] #noffs = knight offsets
boffs = [x for s in [[(i,i),(i,-i),(-i,i),(-i,-i)] for i in range(1,8)] for x in s] # boffs = bishop offsets.
qoffs = boffs + roffs #qoffs = queen offsets.
koffs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)] #koffs = knight offsets

# A move is encoded as (piece_id, offset)
pmoves = [x for s in [[('p'+str(i), pmove) for pmove in poffs] for i in range(8)] for x in s] # 
nmoves = [x for s in [[('n'+str(i), nmove) for nmove in noffs] for i in range(2)]  for x in s]
bmoves = [x for s in [[('b'+str(i), bmove) for bmove in boffs] for i in range(2)]  for x in s]
rmoves = [x for s in [[('r'+str(i), rmove) for rmove in roffs] for i in range(2)]  for x in s]
qmoves = [('q',qmove) for qmove in qoffs]
kmoves = [('k',kmove) for kmove in koffs]
ALL_MOVES = pmoves+nmoves+bmoves+rmoves+qmoves+kmoves

# HELPER
class ValidError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class CaptureError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# add tuples.
def tadd(x, y):
    if len(x) == len(y):
        return tuple([x[i] + y[i] for i in range(len(x))])
    else:
        raise RuntimeError("Can only add tuples with equal length.")

# parse 
def parse_state(fen):
    rows = parseFEN(fen).board.split('\n')[1:9]
    stripped = [r.strip() for r in rows]
    pieces = [[PIECE_DICT[p] for p in list(r)] for r in stripped]
    return np.asarray(pieces)

def matpos_to_fen(rc):
    y,x = rc
    try:
        return XIDX_DICT[x] + YIDX_DICT[y]
    except:
        raise ValidError(rc)

def render_state(fen,c=1):
    pylab.imshow(parse_state(fen).transpose(2,0,1)[c,:,:], interpolation='nearest')

