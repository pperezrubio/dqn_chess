import numpy as np

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

    def __str__(self):
        episodic_str = ''
        if self.episodic:
            episodic_str = 'episodic '
        return "%senvironment with state shape %s and %d actions" %(episodic_str, self.state_shape, self.n_actions)

    def get_state(self):
        raise NotImplementedError("Environment needs get_state().")

    """
    def reward(self, state, action):
        # compute reward given state, action
        raise NotImplementedError("Environment needs reward().")
    """

    def take_action(self, a_idx):
        # alter state 
        # self.step_ctr += 1
        # return reward
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

class Policy(object):
    def __init__(self, environment):
        self.env = environment
        self.state_shape = environment.state_shape
        self.n_actions = environment.n_actions
        
    def action(state):
        raise NotImplementedError("Policy needs mapping from state to action")

def EpsilonGreedy(Policy):
    def __init__(self, epsilon_init, environment):
        super(EpsilonGreedy, self).__init__(environment)
        self.espilon = epsilon_init

    def random_action(self):
        return np.random.randint(self.n_actions)
        
    def greedy_action(self, state):
        raise NotImplementedError("EpsilonGreedy policy needs greedy_action()")
    
    def action(self, state):
        if np.random.binomial(1,self.epsilon):
            return self.random_action()
        else:
            return self.greedy_action()
        
