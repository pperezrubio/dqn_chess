import numpy as np
from experience_replay import Memory, Experience_Replay
from syntaur.models import QNet
from exec import ALL_MOVES

class DQN(object):
    """
    OOP for a Deep Q-Network (DQN). 
    """
    def __init__(self, game, nn_model=None, memory_size = 10000,
                 epsilon = 1, d_rate = 0, batch_size):
        # assumes game is an instance of my_game
        # hence has n_channels, state_shape, num_actions, action_set
        self.memories = ExperienceReplay(memory_size)
        if nn_model is not None:
            self.nnet = nn_model
        self.nn_model = LeNet(game.shape_in, game.n_actions, 
                              nkerns = [16,32],filter_dims=[2,2],
                              fc_dim = 500, out_type = 'linear')
        self.game = game
        self.n_episodes = 0
        self.avg_rewards = []
        self.avg_action_vals = []
        self.epsilon = epsilon
        self.anneal_epsilon = anneal_epsilon

    def _anneal_epsilon(self):
        # handle annealing of exploration rate. 
        pass
            
    def train(n_episodes):
        """
        pseudocode for now.
        """
        # for _ in range(n_episodes):
        #     g = self.game.new_game()
        #     s = g.get_state()
        #     while not g.isTerminal():
        #         if np.random.binomial(n=1,self.epsilon):
        #              a_idx = np.random.randint(self.game.n_actions)
        #              r = g.take_action(a_idx)
        #              stp1 = g.get_state()
        #              self.memories.insert(Memory(s, a_idx, r, stp1))
        #              s = stp1
        #              # compute minibatch update
        #              data = self.memories.sample(batch_size)
        #              data = [memory.target_pair(self.nn_model) for d in data]
        #              self.nn_model.train(data)
        #              # must deal with terminal updates (edge case). 

# Create convolutional neural network. 
# Input is two channel matrix representation of game state
# Output is estimate of value for each action. 
