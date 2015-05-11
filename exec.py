from Chessnut import Game
from Chessnut.game import InvalidMove
from sunfish.xboard import parseFEN
from sunfish.sunfish import search,render
from pprint import pprint
import numpy as np
import string
import pylab

# GLOBAL
PIECE_DICT = {'p':(0,1),'r':(1,1),'n':(2,1),'b':(3,1),'q':(4,1),'k':(5,1),
              'P':(0,0),'R':(1,0),'N':(2,0),'B':(3,0),'Q':(4,0),'K':(5,0),
              '.':(-1,-1)}
XIDX_DICT = {i:c for i,c in zip(range(8),string.lowercase[:8])}
FENX_DICT = {c:i for (i,c) in XIDX_DICT.items()}
YIDX_DICT = {i:j for i,j in zip(range(8),map(str,range(8,0,-1)))}
FENY_DICT = {j:i for (i,j) in YIDX_DICT.items()}

INIT_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

# Assignemnt of id to each piece, keep track of init location
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


# Move encoding
poffs = [(-1,0),(-2,0),(-1,-1),(-1,1)]
roffs = [x for s in [[(-i,0),(i,0),(0,-i),(0,i)] for i in range(1,8)] for x in s]
noffs = [(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)]
boffs = [x for s in [[(i,i),(i,-i),(-i,i),(-i,-i)] for i in range(1,8)] for x in s]
qoffs = boffs + roffs
koffs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
pmoves = [x for s in [[('p'+str(i), pmove) for pmove in poffs] for i in range(8)] for x in s]
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

def tadd(x, y):
    if len(x) == len(y):
        return tuple([x[i] + y[i] for i in range(len(x))])
    else:
        raise RuntimeError("Can only add tuples with equal length.")

def parse_state(g):
    rows = parseFEN(str(g)).board.split('\n')[1:9]
    stripped = [r.strip() for r in rows]
    pieces = [[PIECE_DICT[p] for p in list(r)] for r in stripped]
    return np.asarray(pieces)

def matpos_to_fen(rc):
    y,x = rc
    try:
        return XIDX_DICT[x] + YIDX_DICT[y]
    except:
        raise ValidError(rc)

def fen_to_matpos(fen):
    p1,p2 = fen[:2], fen[2:]
    p1_col,p1_row = p1[0], p1[1]
    p2_col,p2_row = p2[0], p2[1]
    return (FENY_DICT[p1_row], FENX_DICT[p1_col]),(FENY_DICT[p2_row], FENX_DICT[p2_col])
    
def print_board(g):
    print parseFEN(str(g)).board

def render_state(g,c=1):
    pylab.imshow(parse_state(str(g)).transpose(2,0,1)[c,:,:], interpolation='nearest')

def sun_move(g,maxn=20):
    m,_ = search(parseFEN(str(g)).rotate(),20)
    return ''.join(map(render, m))

def only_kings(g):
    if np.sum(np.asarray(g.get_state()[:,:,0] != -1, dtype=int)) == 2:
        return True
    else:
        return False


#def parse_sun_move(tup):


# OOP
class MyGame(object):
    def __init__(self, fen=INIT_FEN):
        self.nmoves = 0
        self.g = Game(fen)
        self.pc_locs = PIECE_LOC.copy()
        self.loc_pcs = LOC_PIECE.copy()
        self.STATE = 0

    def get_state(self):
        rows = parseFEN(str(self.g)).board.split('\n')[1:9]
        stripped = [r.strip() for r in rows]
        pieces = [[PIECE_DICT[p] for p in list(r)] for r in stripped]
        return np.asarray(pieces)

    def __str__(self):
        if self.nmoves %2 == 0:
            return parseFEN(str(self.g)).board
        else:
            return parseFEN(str(self.g)).rotate().board
        
    def render(self,chan=1):
        render_state(self.g,chan)

    def _is_valid(self, fen):
        if fen in self.g.get_moves():
            return True
        else:
            return False

    def _det_move(self, mv_id):
        # returns 
        pc,offset = ALL_MOVES[mv_id]
        pc_loc = self.pc_locs[pc]
        if pc_loc == 'x':
            raise ValidError("Piece %s has been taken." %pc)
        nxt_loc = tadd(pc_loc, offset)
        pc_update = (pc, nxt_loc)
        return ''.join(map(matpos_to_fen, [pc_loc, nxt_loc])), pc_update

    @staticmethod
    def _rand_move():
        return np.random.randint(0,len(ALL_MOVES))
    
    def rand_valid_move(self):
        return np.random.choice(self.g.get_moves())

    def make_move():
        # When does the opponent make a move?
        if _is_valid(fen):
            self.g.apply_move(fen)
        else:
            # penalize. and prompt to try again. 
            pass

    def move(self,fen):
        try:
            self.g.apply_move(fen)
            self.nmoves += 1
        except e:
            return e

    def step(self):
        try:
            fen,update = self._det_move(rand_move())
            if not self._is_valid(fen):
                raise RuntimeError("Move %s not valid." %fen)
            pc,nxtloc = update
            inloc = self._pc_locs[pc]
            print "randomly chosen action go!"
            print fen
        except:
#            print "randomly chosen from valid actions go!"
            fen = np.random.choice(self.g.get_moves())
#            print fen
            inloc, nxtloc = fen_to_matpos(fen)
            pc = self.loc_pcs[inloc]
            update = (pc, nxtloc)

        # Carry out move. 
        # Update state for where each of our pieces is. 
        self.g.apply_move(fen)
        self.nmoves += 1
        self.loc_pcs.pop(inloc, None)
        self.pc_locs[pc] = nxtloc
        self.loc_pcs[nxtloc] = pc
        # Opponent makes move. 
        #        self.g.apply_move(sun_move(self.g))
        # Opponent chooses randomly
        try:
            self.g.apply_move(self.rand_valid_move())
            self.nmoves += 1
        except:
            return # no moves available. 
        # See if any of our pieces have been taken. 
        s = self.get_state()
        for (p,l) in self.pc_locs.items():
            if l != 'x':
                if s[l][1] != 0:
                    print "%s captured at %s" %(str(p), str(l))
                    self.pc_locs[p] = 'x'
                    self.loc_pcs.pop(l,None)

    def play_game(self):
        i = 0
        states = []
        while not self.STATE in [2,3]:
            if i%20 == 0:
                print "step %d" %i
            if i%50 == 0:
                if only_kings(self):
                    self.STATE = 3
            states.append(self.get_state())
            self.step()
            i += 1
            if self.STATE != 3:
                self.STATE = self.g.status
        return s

# MAIN
g = MyGame()
