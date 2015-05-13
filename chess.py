from rl import Episodic
from execution import ALL_MOVES, PIECE_LOC, LOC_PIECE, PIECE_DICT, tadd, matpos_to_fen, CaptureError
from Chessnut import Game
from sunfish.xboard import parseFEN, mrender
from sunfish.sunfish import search
import numpy as np

# formulate chess game as Episodic environment. 
class ChessGame(Episodic):
    def __init__(self):
        super(ChessGame, self).__init__((8,8,2),224)
        self.game = Game()
        self.INVALID_MOVE_REWARD = -1
        self.LOSS_REWARD = -10
        self.DRAW_REWARD = -5
        self.WIN_REWARD = 10
        self._pc_locs = PIECE_LOC.copy()
#        self._loc_pcs = LOC_PIECE.copy()
        self._num_moves = 0
        self._STATUS = 0

    def __str__(self):
        status_str = '' 
        if self._STATUS == 0:
            status_str = ', in_progress.'
        elif self._STATUS == 1:
            status_str = ', %s in check' %self.g.state.player
        elif self._STATUS == 2:
            status_str = ', %s in checkmate' %self.g.state.player
        elif self._STATUS == 3:
            status_str = ', ended in draw.' 
        else:
            raise RuntimeError("Internal Error. STATUS not in [0-3]")
        return "Chess game%s" %status_str

    def _print_board(self):
        if self._num_moves %2 == 0:
            print parseFEN(str(self.game)).board
        else:
            print parseFEN(str(self.game)).rotate().board
    
    # determine the move (in FEN notation) corresponding to action index
    # given current board state. 
    def _det_move(self, a_idx):
        piece, offset = ALL_MOVES[a_idx]
        piece_location = self._pc_locs[piece]
        if piece_location == 'x':
            raise CaptureError("Piece %s has been captured." %piece)
        nxt_location = tadd(piece_location, offset)
        return ''.join(map(matpos_to_fen, [piece_location, nxt_location]))

    def _is_valid(self, fen_move):
        return fen_move in self.game.get_moves()

    # update internal dictionaries mapping piece to board location
    # and mapping board location to piece.
    def _update_pc_locs(self, a_idx):
        piece, offset = ALL_MOVES[a_idx]
        piece_location = self._pc_locs[piece]
        nxt_location = tadd(piece_location, offset)
        self._pc_locs[piece] = nxt_location

    # update piece location dict if a piece was taken.
    def _update_if_taken(self):
        for (piece, loc) in self._pc_locs.items():
            # if a piece hasn't been taken already, ...
            if loc != 'x':
                state = self.get_state()
                if state[loc][1] != 0:
                    self._pc_locs[piece] = 'x'

    def _only_kings(self):
        return np.sum(np.asarray(self.get_state()[:,:,0]!=-1, dtype=int))==2
        
    def _sunfish_move(self, search_depth = 20):
        color = 0 if self.game.state.player == 'w' else 1
        pos = parseFEN(str(self.game))
        m,_ = search(pos, search_depth)
        return mrender(color, pos, m)

    # just for debugging.
    def _move(self, fen):
        self.game.apply_move(fen)
        self._num_moves += 1

    def get_state(self):
        rows = parseFEN(str(self.game)).board.split('\n')[1:9]
        stripped = [r.strip() for r in rows]
        pieces = [[PIECE_DICT[p] for p in list(r)] for r in stripped]
        return np.asarray(pieces)

    def is_terminal(self, state):
        if self._only_kings():
            self._STATUS = 3
        return self._STATUS not in [0,1]

    def take_action(self, a_idx):
        self.step_ctr += 1
        try:
            move = self._det_move(a_idx)
        except CaptureError:
            reward = self.INVALID_MOVE_REWARD
            self.episode_rewards.append(reward)
            return reward
        except:
            raise RuntimeError("Internal error: _det_move failed on %d" %a_idx)
        if not self._is_valid(move):
            reward = self.INVALID_MOVE_REWARD
            self.episode_rewards.append(reward)
            return reward
        else:
            # update state (carry out move)
            self.game.apply_move(move)
            # increment move counter
            self._num_moves += 1 
            # compute location updates. 
            self._update_pc_locs(a_idx)
            # carry out opponent move
            bot_move = self._sunfish_move()
            self.game.apply_move(bot_move)
            # increment move counter
            self._num_moves += 1
            # check if pieces were taken
            self._update_if_taken()
            # update status (i.e. check for checkmate)
            self._STATUS = self.game.status
            # check if state is terminal.
            if self.is_terminal(self.get_state()):
                if self._STATUS == 3: # draw
                    reward = self.DRAW_REWARD
                    self.episode_rewards.append(reward)
                    return reward
                elif self._STATUS == 2: #checkmate
                    if self.game.state.player == 'w':
                        reward = self.LOSS_REWARD
                        self.episode_rewards.append(reward)
                        return reward
                    else:
                        reward = self.WIN_REWARD
                        self.episode_rewards.append(reward)
                        return reward
                else:
                    raise RuntimeError("Internal error: invalid status.")
            else:
                reward = 0 # non-terminal, valid action.
                self.episode_rewards.append(reward)
                return reward
    
    def reset(self):
        # reset state variables
        self._pc_locs = PIECE_LOC.copy()
        self._num_moves = 0
        self._STATUS = 0
        # new game instance. 
        self.game = Game()
        self.avg_rewards.append(np.mean(self.episode_rewards))
        self.episode_rewards = []
        self.episode_ctr += 1
