import numpy as np
from experience_replay import Memory, ExperienceReplay
from syntaur.models import LeNet
from syntaur.optimize import single_batch_trainer
from execution import ALL_MOVES
from chess import ChessGame
import matplotlib.pyplot as plt

# TODO - write game method to say how game ended.

class DQN(object):
    """
    OOP for a Deep Q-Network (DQN). 
    """
    def __init__(self, game, memory_size = 100000, 
                 batch_size = 1, epsilon_init = 1.0, alpha_init = .00025,
                 anneal_alpha = True, anneal_epsilon = True, 
                 batch_size_incr = 0):
        self.memories = ExperienceReplay(memory_size)
        self.nnet = LeNet(game.state_shape, dim_out = game.n_actions, 
                          batch_size = 1, fc_dim = 500,
                          nkerns = [16,32], filter_dims = [2,2],
                          out_type = 'linear')
        self.trainer = single_batch_trainer(self.nnet)
        self.game = game
        self.n_episodes = 0
        self.avg_rewards = []
        self.avg_action_vals = []
        self.alpha = alpha_init
        self.epsilon = epsilon_init
        self.anneal_ep = anneal_epsilon
        self.anneal_lr = anneal_alpha
        self.batch_size = batch_size
        self.batch_size_incr = batch_size_incr
        self._pct_invalids = []
        self._costs = []
            
    def train(self, n_episodes = 3, max_iter = 500):
        g = self.game
        g.reset()
        # set anneal rate for epsilon.
        ep_anneal_rate = 0
        if self.anneal_ep:
            ep_anneal_rate = self.epsilon/n_episodes
        alpha_anneal_rate = 0
        if self.anneal_lr:
            alpha_anneal_rate = self.alpha/n_episodes
        for e_idx in range(n_episodes):
            s = g.get_state()
            print "Episode: %d, Exploration Rate: %f, Learning Rate: %f" %(e_idx, self.epsilon, self.alpha)
            while not g.is_terminal() and not self.game._num_moves >= max_iter and not self.game.iter_ctr >= 200:
                if np.random.binomial(1,self.epsilon):
                    a_idx = np.random.randint(self.game.n_actions)
                else:
                    values = self.nnet.outputter(s.reshape(self.nnet.image_shape))
                    a_idx = np.argmax(values[0])
                r = g.take_action(a_idx)
                stp1 = g.get_state()
                self.memories.insert(Memory(
                    s.transpose(2,0,1).reshape(self.nnet.image_shape), 
                    a_idx, 
                    r, 
                    stp1.transpose(2,0,1).reshape(self.nnet.image_shape)
                ))
                s = stp1

                # TEST CLOOJ
                if self.game.iter_ctr %200 == 0:
                    print "move_n: %d, action: %d, reward: %d, status: %d" %(
                        self.game.iter_ctr, a_idx, r, self.game._STATUS
                    )
                
                # TODO - compute minibatch update
                # //pseudocode
                if e_idx > 0:
                    costs = []
                    data = self.memories.sample(self.batch_size)
                    data = [m.target_pair(self.nnet) for m in data]
                    for i in range(self.batch_size):
                        d = data[i]
                        costs.append(self.trainer(d[0], d[1], self.alpha))
                    if self.game.iter_ctr%10 == 0:
                        print "cost: %f" %(np.mean(costs))
                        self._costs.append(np.mean(costs))
                    if self.game.iter_ctr%500 == 0:
                        self.game._print_board()
            print "Game %d ends in %d iterations with status %d, reward %d." %(e_idx, self.game.iter_ctr, self.game._STATUS, r)
            # compute percent invalid actions.
            n_moves = g.iter_ctr
            rs = g.episode_rewards
            n_invalid = len(np.where(rs == np.array([-.02 for _ in range(len(rs))]))[0])
            pct_invalid = float(n_invalid)/n_moves
            self._pct_invalids.append(pct_invalid)
            print "Pct Invalid: %f" %pct_invalid
            g.reset()
            self.epsilon -= ep_anneal_rate
            self.batch_size += self.batch_size_incr
            if e_idx > 0:
                self.alpha -= alpha_anneal_rate

        

# Create convolutional neural network. 
# Input is two channel matrix representation of game state
# Output is estimate of value for each action. 

game = ChessGame()
dqn = DQN(game)
dqn.train(1000,50)
plt.plot(dqn._pct_invalids)
plt.figure()
plt.plot(dqn._costs)
dqn.game.reset()
dqn._pct_invalids = []
dqn._costs = []
dqn.batch_size = 5
dqn.epsilon = 0
dqn.alpha = .00025
dqn.game.search_depth = 5
dqn.train(50,5)


"""
dqn.batch_size_incr = 0
dqn.alpha = .1
dqn.epsilon = .8
dqn.train(1,1000)
dqn.alpha = .2
dqn.epsilon = .48
dqn.train(40,1000)
dqn.alpha = .1
dqn.epsilon = .5
dqn.game.search_depth = 5
dqn.train(20,10000)
dqn.alpha = .1
dqn.epsilon = .5
dqn.game.search_depth = 20
dqn.train(20,10000)
dqn.alpha = .1
dqn.epsilon = .5
dqn.game.search_depth = 20
dqn.train(20,10000)
"""

"""
dqn.alpha = .1
dqn.epsilon = .2
dqn.train(20,10000)
dqn.alpha = .1
dqn.epsilon = .2
dqn.train(20,10000)
dqn.alpha = .05
dqn.epsilon = .05
"""    
