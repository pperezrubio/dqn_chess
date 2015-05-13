import numpy as np
from experience_replay import Memory, ExperienceReplay
#from syntaur.models import QNet
from execution import ALL_MOVES
from chess import ChessGame

# TODO - write game method to say how game ended.

class DQN(object):
    """
    OOP for a Deep Q-Network (DQN). 
    """
    def __init__(self, game, memory_size = 100000, 
                 d_rate = 0, batch_size = 50, epsilon = 1, nn_model=None):
        # assumes game is an instance of my_game
        # hence has n_channels, state_shape, num_actions, action_set
        self.memories = ExperienceReplay(memory_size)
        if nn_model is not None:
            self.nnet = nn_model
#        self.nn_model = LeNet(game.shape_in, game.n_actions, 
#                              nkerns = [16,32],filter_dims=[2,2],
#                              fc_dim = 500, out_type = 'linear')
        self.game = game
        self.n_episodes = 0
        self.avg_rewards = []
        self.avg_action_vals = []
        self.epsilon = epsilon
#        self.anneal_epsilon = anneal_epsilon

#    def _anneal_epsilon(self):
        # TODO - handle annealing of exploration rate. 
#        pass
            
    def train(self, n_episodes = 3, max_iter = 500):
        g = self.game
        for e_idx in range(n_episodes):
            s = g.get_state()
            print "episode: %d" %e_idx
            while not g.is_terminal() and not self.game.iter_ctr >= max_iter:
                if np.random.binomial(1,self.epsilon):
                    a_idx = np.random.randint(self.game.n_actions)
                else:
                    a_idx = self.game._rand_valid()
                    # TODO - a_idx = greedy action
                r = g.take_action(a_idx)
                stp1 = g.get_state()
                self.memories.insert(Memory(s, a_idx, r, stp1))
                s = stp1

                # TEST CLOOJ
                if self.game.iter_ctr%50 == 0:
                    print "move_n: %d, action: %d, reward: %d, status: %d" %(
                        self.game.iter_ctr, a_idx, r, self.game._STATUS
                    )
                
                # TODO - compute minibatch update
                # //pseudocode
                # data = self.memories.sample(batch_size)
                # data = [memory.target_pair(self.nn_model) for d in data]
                # self.nn_model.train(data)
                # // end pseudocode
                # TODO - must deal with terminal updates (edge case). 
            print "Game %d ends with status %d." %(e_idx, self.game._STATUS)
            g.reset()

        

# Create convolutional neural network. 
# Input is two channel matrix representation of game state
# Output is estimate of value for each action. 

def test():
    dqn = DQN(ChessGame())
    dqn.train()
    
