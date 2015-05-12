class Environment(object):
    # OOP for an 'environment' in the reinforcement learning sense. 
    def __init__(self, state_shape, n_actions, episodic = True):
        """
        state_shape is a tuple of ints, e.g. (xdim, ydim, n_channels)
        n_actions is an integer, supposes the action space is constant.
        """
        self.step_ctr = 0
        self.episodic = episodic
        self.state_shape = state_shape
        self.n_actions = n_actions
        # initialize state

    def get_state(self):
        raise NotImplementedError("Environment needs get_state().")

    def reward(self, state, action):
        # compute reward given state, action
        raise NotImplementedError("Environment needs reward().")

    def take_action(self, a_idx):
        # alter state 
        # return reward
        # self.step_ctr += 1
        raise NotImplementedError("Environment needs take_action().")

class Episodic(Environment):
    def __init__(self, state_shape, n_actions):
        super(Episodic, self).__init__(state_shape, n_actions, episodic=True)
        self.episode_ctr = 0
        self.episode_rewards = []
        self.avg_rewards = []

    def reset(self):
        # average episode rewards and append to avg_rewards
        # clear episode rewards
        # increment episode_ctr
        # return to init_state
        raise NotImplementedError("Episodic task needs reset().")

    def is_terminal(self, state):
        raise NotImplementedError("Episodic task needs is_terminal().")
        


        
