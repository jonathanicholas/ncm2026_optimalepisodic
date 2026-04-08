import numpy as np
import random
from scipy.stats import norm

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete




class MemoryAcceptLeaveEnv(gym.Env):
    """
    A memory-guided accept/leave decision-making environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            num_slots = 6,
            num_items = 16,
            num_features = 4,
            value_min = -9,
            value_max = 9,
            reward_std = 3.,
            relevance_prob = 0.8,
            init_num_items = 6,
            t_max = 100,
            stay_cost = 0.008,
            saccade_cost = 0.04,
            scale_factor = 1 / 5,
            mode = 'random',
            seed = None,
        ):
        """
        Construct an environment.
        """

        # set random seed
        self.set_random_seed(seed)

        # initialize parameters
        self.num_slots = num_slots # number of slots
        self.num_items = num_items # number of items
        self.num_features = num_features # number of features
        self.value_min = value_min # min values
        self.value_max = value_max # max value
        self.value_set = np.arange(self.value_min, self.value_max + 1) # set of values
        self.value_set = self.value_set[self.value_set != 0]
        self.reward_std = reward_std # std of reward observation
        self.relevance_prob = relevance_prob # probability of relevance observation
        self.init_num_items = int(np.clip(init_num_items, 0, self.num_slots)) # number of initial informed items
        self.t_max = t_max # max time steps per episode
        self.stay_cost = stay_cost # stay cost
        self.saccade_cost = saccade_cost # saccade cost
        self.scale_factor = scale_factor # reward scale factor
        self.mode = mode # environment mode

        # initialize items
        self.feature_set = np.array([int_to_bits(item, n_bits = self.num_features) for item in np.arange(0, self.num_items)]) # (num_items, num_features)

        # initialize all pre-generated consitions
        self.all_items = np.array(ALL_ITEMS)
        self.all_values = np.array(ALL_VALUES)

        # initialize action space
        self.action_space = Discrete(self.num_slots + 2)

        # initialize observation space
        observation_shape = (
            self.num_slots + # fixation slot (num_slots,)
            3 + # time, reward observation, relevance observation
            2 * self.num_slots, # initial samples
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = observation_shape,)


    def reset(self):
        """
        Reset the environment.
        """

        # reset the trial
        self.init_trial()

        # get observation
        obs = self.get_obs()
    
        # get info
        info = {
            'mask': self.get_action_mask(),
        }
        return obs, info


    def step(self, action):
        """
        Step the environment.
        """

        # make sure int
        action = int(action)

        self.time_elapsed += 1
        done = False
        reward = 0. # initialize reward as 0

        # if fixation
        if action < self.num_slots:
            prev_slot = self.fixation_slot

            # compute distance
            if prev_slot is None:
                distance = 0
            else:
                distance = self.saccade_distance(prev_slot, action)
            
            reward -= (self.stay_cost + self.saccade_cost * distance) * self.scale_factor

            # move fixation to the slot
            self.fixation_slot = action

            # clear reward and relevance observation
            reward_sample, relevance_sample = self.sample_reward_and_relevance_at_slot(self.fixation_slot)
            self.reward_obs = reward_sample
            self.relevance_obs = relevance_sample
        
        # if decision
        elif action >= self.num_slots:
            # remove fixation slot
            self.fixation_slot = None

            # accept
            if action == self.num_slots:
                reward += self.offer_value * self.scale_factor

            # leave
            elif action == self.num_slots + 1:
                pass

            else:
                raise ValueError('Action index exceeded.')

        # if make a decision within time limit
        if action >= self.num_slots or self.time_elapsed == self.t_max:
            done = True

        # get observation
        obs = self.get_obs()
    
        # get info
        info = {
            'mask': self.get_action_mask(),
        }

        return obs, reward, done, False, info
    
    
    def init_trial(self):
        """
        Initialize a trial.
        """

        # initialize time_elapsed
        self.time_elapsed = 0

        # randomly generate items and values
        if self.mode == 'random':

            # sample items
            while True:
                # try a sample of items
                items = np.random.choice(np.arange(self.num_items), size = self.num_slots, replace = False)

                # count how many relevant items each feature would produce
                counts = self.feature_set[items].sum(axis = 0) # (num_features,)

                # find features giving exactly half relevant items
                eligible_features = np.where(counts == int(self.num_slots / 2))[0]

                # if found, accept this trial
                if eligible_features.size > 0:
                    # get items
                    self.items = items

                    # compute query feature
                    self.query_feature = int(np.random.choice(eligible_features))

                    break

            # sample values
            self.values = np.random.choice(self.value_set, size = self.num_slots, replace = True) # (num_slots,)

        # use pre-generated conditions to generate items and values
        elif self.mode == 'exp':
            # randomly sample a condition and a permutation
            index = np.random.randint(0, len(self.all_items))
            perm = np.random.permutation(self.num_slots)

            # compute items
            self.items = self.all_items[index][perm]

            # compute values
            self.values = self.all_values[index][perm]

            # sample query feature
            self.query_feature = np.random.randint(0, self.num_features) # (1,)
        
        else:
            raise ValueError('Mode must be random or exp.')

        # compute relevance
        self.relevances = self.feature_set[self.items, self.query_feature] # (num_slots,), binary

        # compute offer value
        self.offer_value = np.sum(self.values * self.relevances) # (1,)

        # initialize fixation slot
        self.fixation_slot = None

        # initialize observations
        self.reward_obs = 0.0
        self.relevance_obs = 0.5

        # initialize initial samples
        self.init_reward_obs = np.zeros((self.num_slots,), dtype = float)
        self.init_relevance_obs = np.full((self.num_slots,), 0.5, dtype = float)

        # choose which slots to reveal at time 0
        if self.init_num_items > 0:
            revealed_slots = np.random.choice(self.num_slots, size = self.init_num_items, replace = False)

            for slot in revealed_slots:
                reward_sample, relevance_sample = self.sample_reward_and_relevance_at_slot(slot)
                self.init_reward_obs[slot] = reward_sample
                self.init_relevance_obs[slot] = relevance_sample


    def sample_reward_and_relevance_at_slot(self, slot):
        """
        Sample a reward and a noisy relevance observation for a given slot.
        Returns (reward_sample, relevance_sample).
        """

        # reward observation (Gaussian)
        reward_sample = np.random.normal(loc = self.values[slot], scale = self.reward_std)

        # relevance observation (Bernoulli with reliability)
        true_relevance = self.relevances[slot]
        p_correct = self.relevance_prob

        # P(obs=1)
        p_obs_1 = true_relevance * p_correct + (1 - true_relevance) * (1 - p_correct)
        relevance_sample = np.random.binomial(1, p_obs_1)

        return reward_sample, relevance_sample
    

    def saccade_distance(self, slot1, slot2):
        """
        Circular distance on a ring of size num_slots.
        """

        diff = abs(slot1 - slot2)

        return min(diff, self.num_slots - diff)
    

    def get_obs(self):
        """
        Get observation.
        """

        # original part
        base = np.hstack([
            self.one_hot_coding(num_classes = self.num_slots, labels = self.fixation_slot),
            self.reward_obs,
            self.relevance_obs,
            float(self.time_elapsed),
        ])

        # extra 12 channels: only informative at time 0
        if self.time_elapsed == 0:
            init_part = np.hstack([self.init_reward_obs, self.init_relevance_obs])
        else:
            init_part = np.hstack([
                np.zeros((self.num_slots,), dtype = float), # reward init channels -> 0
                np.full((self.num_slots,), 0.5, dtype = float), # relevance init channels -> 0.5
            ])

        # wrap observation
        obs = np.hstack([base, init_part])

        return obs
    

    def get_action_mask(self):
        """
        Get action mask.

        Note:
            no batching is considered here. batching is implemented by vectorzation wrapper.
            if no batch training is used, add the batch dimension and transfer the mask to torch.tensor in trainer.
            if batch training is used, concatenate batches and transfer the mask to torch.tensor in trainer.
        """

        mask = np.ones((self.action_space.n,), dtype = bool)
        
        return mask


    def set_random_seed(self, seed):
        """
        Set random seed.
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)


    def one_hot_coding(self, num_classes, labels = None):
        """
        One-hot code nodes.
        """

        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]

        return labels_one_hot


class MetaLearningWrapper(Wrapper):
    """
    A meta-RL wrapper.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, env):
        """
        Construct an wrapper.
        """

        super().__init__(env)

        self.env = env
        self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')

        # initialize previous variables
        self.init_prev_variables()

        # define new observation space
        new_observation_shape = (
            self.env.observation_space.shape[0] + # obs
            self.env.action_space.n + # previous action
            1, # previous reward
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = new_observation_shape)


    def step(self, action):
        """
        Step the environment.
        """

        obs, reward, done, truncated, info = self.env.step(action)

        # concatenate previous variables into observation
        obs_wrapped = self.wrap_obs(obs)

        # update previous variables
        self.prev_action = action
        self.prev_reward = reward

        return obs_wrapped, reward, done, truncated, info
    

    def reset(self, seed = None, options = {}):
        """
        Reset the environment.
        """

        obs, info = self.env.reset()

        # initialize previous physical action and reward
        self.init_prev_variables()

        # concatenate previous physical action and reward into observation
        obs_wrapped = self.wrap_obs(obs)

        return obs_wrapped, info
    

    def init_prev_variables(self):
        """
        Reset previous variables.
        """

        self.prev_action = None
        self.prev_reward = 0.


    def wrap_obs(self, obs):
        """
        Wrap observation with previous variables.
        """

        obs_wrapped = np.hstack([
            obs, # current obs
            self.one_hot_coding(num_classes = self.env.action_space.n, labels = self.prev_action),
            self.prev_reward,
        ])
        return obs_wrapped


def int_to_bits(x, n_bits):
    """
    Transform number into bits
    """
    return ((x >> np.arange(n_bits)) & 1).astype(int)
    

ALL_ITEMS = [
    [15, 2, 0, 8, 7, 13],
    [0, 6, 11, 5, 14, 9],
    [1, 4, 3, 15, 12, 10],
    [15, 7, 2, 0, 8, 13],
    [6, 14, 5, 0, 11, 9],
    [12, 1, 3, 10, 15, 4],
    [8, 1, 6, 7, 10, 13],
    [2, 5, 0, 15, 10, 13],
    [7, 6, 14, 8, 1, 9],
    [11, 12, 5, 4, 10, 3],
    [13, 10, 5, 2, 15, 0],
    [8, 14, 9, 7, 6, 1],
    [5, 12, 11, 4, 10, 3],
    [14, 8, 7, 6, 1, 9],
    [4, 6, 11, 9, 15, 0],
    [8, 2, 3, 13, 12, 7],
    [5, 1, 0, 15, 14, 10],
    [4, 9, 11, 0, 6, 15],
    [13, 12, 3, 2, 8, 7],
    [0, 15, 1, 14, 10, 5],
    [11, 2, 13, 8, 7, 4],
    [13, 8, 7, 6, 9, 2],
    [9, 3, 12, 10, 5, 6],
    [1, 15, 11, 4, 14, 0],
    [13, 9, 8, 6, 2, 7],
    [5, 10, 6, 9, 3, 12],
    [0, 14, 1, 11, 15, 4],
    [15, 12, 4, 10, 3, 1],
    [1, 15, 8, 12, 2, 7],
    [13, 3, 12, 2, 6, 9],
    [4, 5, 10, 15, 11, 0],
    [8, 12, 2, 1, 7, 15],
    [13, 9, 6, 12, 3, 2],
    [5, 11, 0, 10, 4, 15],
    [8, 4, 11, 3, 5, 14],
    [5, 11, 10, 14, 4, 1],
    [9, 3, 7, 12, 0, 14],
    [11, 5, 2, 6, 12, 9],
    [4, 1, 14, 5, 11, 10],
    [3, 8, 15, 4, 13, 2],
    [2, 6, 5, 11, 12, 9],
    [3, 14, 12, 7, 9, 0],
    [3, 13, 4, 12, 2, 11],
    [4, 15, 13, 3, 10, 0],
    [10, 3, 12, 1, 6, 13],
    [2, 5, 9, 14, 7, 8],
    [10, 13, 15, 3, 4, 0],
    [11, 4, 1, 14, 8, 7],
    [8, 9, 5, 14, 7, 2],
    [8, 10, 4, 5, 7, 11],
    [15, 14, 12, 0, 1, 3],
    [2, 1, 6, 14, 13, 9],
    [8, 4, 11, 5, 7, 10],
    [3, 0, 15, 12, 1, 14],
    [9, 14, 2, 6, 13, 1],
    [5, 10, 4, 7, 8, 11],
    [0, 7, 1, 14, 13, 10],
    [5, 9, 6, 2, 15, 8],
    [15, 13, 0, 11, 2, 4],
    [0, 10, 12, 7, 13, 3],
    [8, 14, 6, 3, 9, 5],
    [11, 4, 15, 2, 13, 0],
    [1, 6, 5, 8, 15, 10],
    [4, 11, 2, 12, 3, 13],
    [10, 9, 5, 7, 8, 6],
    [2, 15, 0, 13, 1, 14],
    [3, 11, 2, 12, 13, 4],
    [10, 7, 9, 8, 5, 6],
    [13, 2, 15, 1, 0, 14],
    [5, 10, 8, 7, 9, 6],
    [4, 15, 5, 10, 0, 11],
    [12, 8, 7, 3, 13, 2],
    [6, 2, 14, 9, 1, 13],
    [15, 4, 11, 0, 10, 5],
    [7, 13, 2, 3, 8, 12],
    [1, 9, 6, 13, 14, 2],
    [15, 11, 4, 10, 0, 5],
    [3, 9, 4, 14, 0, 15],
    [3, 4, 0, 11, 12, 15],
    [15, 7, 1, 8, 0, 14],
    [6, 10, 5, 13, 2, 9],
    [12, 15, 3, 0, 11, 4],
    [5, 9, 10, 6, 13, 2],
    [5, 6, 9, 13, 2, 10],
    [7, 5, 0, 9, 14, 10],
    [7, 2, 8, 13, 9, 6],
    [12, 3, 11, 4, 13, 2],
    [11, 4, 1, 10, 5, 14],
    [5, 15, 0, 7, 8, 10],
    [2, 13, 4, 12, 11, 3],
    [5, 11, 14, 1, 10, 4],
    [14, 4, 11, 10, 5, 1],
    [2, 9, 13, 12, 3, 6],
    [0, 5, 10, 15, 14, 1],
    [6, 10, 4, 11, 5, 9],
    [8, 7, 2, 15, 1, 12],
    [0, 10, 5, 15, 14, 1],
    [6, 3, 9, 2, 13, 12],
    [8, 11, 6, 4, 9, 7],
    [1, 5, 14, 13, 10, 2],
    [0, 12, 3, 10, 5, 15],
    [6, 4, 8, 7, 9, 11],
    [14, 2, 13, 1, 5, 10],
    [5, 12, 3, 0, 10, 15],
    [11, 7, 6, 8, 4, 9],
    [11, 12, 4, 8, 7, 3],
    [0, 15, 1, 9, 6, 14],
    [2, 13, 5, 15, 0, 10],
    [11, 7, 4, 12, 3, 8],
    [6, 9, 14, 1, 15, 0],
    [5, 2, 0, 13, 15, 10],
    [12, 8, 3, 4, 11, 7],
    [5, 1, 14, 10, 15, 0],
    [2, 7, 10, 5, 13, 8],
    [4, 3, 11, 9, 6, 12],
    [15, 0, 14, 10, 5, 1],
    [8, 5, 7, 10, 13, 2],
    [4, 6, 9, 12, 11, 3],
    [11, 3, 4, 6, 9, 12],
    [5, 14, 1, 2, 13, 10],
    [3, 4, 8, 7, 11, 12],
    [6, 4, 11, 7, 9, 8],
    [15, 5, 1, 10, 14, 0],
    [7, 4, 8, 12, 11, 3],
    [7, 6, 8, 4, 11, 9],
    [2, 10, 1, 5, 13, 14],
    [3, 7, 13, 12, 2, 8],
    [9, 6, 15, 0, 1, 14],
    [8, 3, 4, 11, 12, 7],
    [7, 12, 2, 3, 13, 8],
    [10, 5, 0, 9, 6, 15],
    [8, 11, 3, 7, 4, 12],
    [9, 0, 1, 14, 6, 15],
    [5, 6, 13, 10, 9, 2],
    [8, 4, 6, 3, 9, 15],
    [1, 11, 14, 12, 0, 7],
    [5, 13, 9, 10, 2, 6],
    [9, 6, 15, 3, 8, 4],
    [11, 7, 14, 0, 12, 1],
    [12, 2, 0, 7, 11, 13],
    [1, 7, 12, 14, 8, 3],
    [9, 6, 13, 4, 11, 2],
    [15, 0, 10, 4, 11, 5],
    [3, 12, 8, 1, 14, 7],
    [2, 11, 9, 6, 4, 13],
    [11, 0, 15, 4, 10, 5],
    [8, 7, 1, 14, 12, 3],
    [0, 12, 13, 3, 2, 15],
    [14, 10, 1, 4, 9, 7],
    [5, 8, 1, 11, 14, 6],
    [0, 3, 15, 13, 2, 12],
    [7, 4, 9, 10, 1, 14],
    [5, 1, 11, 14, 8, 6],
    [3, 13, 0, 2, 15, 12],
    [12, 14, 0, 1, 15, 3],
    [0, 8, 7, 15, 6, 9],
    [15, 2, 13, 4, 1, 10],
    [12, 14, 1, 0, 15, 3],
    [11, 10, 5, 4, 6, 9],
    [11, 1, 14, 5, 10, 4],
    [5, 6, 9, 4, 10, 11],
    [5, 3, 10, 14, 1, 12],
    [8, 7, 3, 2, 12, 13],
    [15, 0, 9, 4, 11, 6],
    [12, 1, 3, 10, 14, 5],
    [2, 13, 3, 7, 8, 12],
    [11, 0, 6, 15, 4, 9],
    [6, 5, 9, 14, 0, 11],
    [3, 2, 12, 0, 13, 15],
    [11, 6, 3, 4, 8, 13],
    [9, 7, 13, 0, 6, 10],
    [0, 2, 15, 3, 13, 12],
    [1, 5, 14, 11, 10, 4],
    [4, 11, 5, 1, 14, 10],
    [11, 5, 10, 1, 14, 4],
    [3, 14, 4, 8, 5, 11],
    [6, 9, 4, 8, 11, 7],
    [15, 4, 0, 7, 11, 8],
    [3, 12, 4, 10, 5, 11],
    [2, 9, 14, 1, 13, 6],
    [15, 2, 13, 0, 14, 1],
    [13, 2, 14, 1, 6, 9],
    [9, 14, 4, 2, 3, 13],
    [6, 12, 1, 5, 11, 10],
    [3, 8, 15, 0, 12, 7],
    [3, 9, 14, 4, 2, 13],
    [10, 1, 6, 5, 12, 11],
    [8, 0, 15, 3, 12, 7],
    [0, 5, 15, 14, 9, 2],
    [12, 8, 10, 7, 3, 5],
    [7, 2, 11, 12, 13, 0],
    [6, 15, 9, 13, 2, 0],
    [10, 7, 8, 12, 3, 5],
    [0, 11, 15, 14, 1, 4],
    [9, 0, 2, 15, 13, 6],
    [3, 5, 8, 10, 7, 12],
    [6, 8, 13, 11, 2, 5],
    [11, 7, 9, 14, 0, 4],
    [8, 1, 3, 15, 12, 6],
    [2, 9, 10, 6, 13, 5],
    [11, 14, 4, 0, 7, 9],
    [12, 8, 3, 15, 6, 1],
    [4, 7, 0, 9, 14, 11],
    [6, 8, 11, 0, 5, 15],
    [12, 3, 13, 2, 7, 8],
    [2, 14, 1, 9, 4, 15],
    [8, 5, 6, 11, 0, 15],
    [12, 7, 8, 2, 3, 13],
    [5, 10, 14, 3, 9, 4],
    [15, 14, 4, 9, 1, 2],
    [12, 14, 3, 0, 1, 15],
    [7, 11, 5, 10, 8, 4],
    [11, 3, 12, 6, 0, 13],
    [3, 1, 12, 0, 14, 15],
    [5, 12, 9, 10, 3, 6],
    [2, 4, 15, 0, 13, 11],
    [9, 10, 4, 2, 15, 5],
    [15, 9, 7, 0, 8, 6],
    [0, 11, 14, 4, 1, 15],
    [3, 13, 5, 2, 12, 10],
    [7, 8, 9, 0, 15, 6],
    [1, 4, 11, 14, 0, 15],
    [3, 5, 12, 10, 13, 2],
    [10, 9, 13, 3, 6, 4],
    [14, 10, 1, 11, 5, 4],
    [8, 9, 12, 3, 6, 7],
    [14, 1, 0, 13, 15, 2],
    [5, 4, 1, 11, 10, 14],
    [8, 7, 6, 12, 9, 3],
    [14, 13, 15, 0, 2, 1],
    [6, 7, 9, 8, 12, 3],
    [14, 8, 1, 4, 11, 7],
    [8, 4, 3, 15, 13, 2],
    [2, 5, 13, 3, 10, 12],
    [8, 1, 7, 4, 11, 14],
    [10, 0, 6, 15, 9, 5],
    [5, 12, 13, 2, 3, 10],
    [7, 4, 11, 14, 8, 1],
    [6, 11, 0, 15, 12, 1],
    [4, 14, 10, 3, 13, 1],
    [6, 9, 4, 11, 7, 8],
    [10, 5, 12, 2, 9, 7],
    [1, 4, 14, 10, 13, 3],
    [7, 8, 6, 9, 11, 4],
    [0, 15, 12, 1, 11, 6],
    [10, 5, 15, 9, 0, 6],
    [2, 11, 4, 7, 13, 8],
    [14, 1, 9, 12, 6, 3],
    [0, 9, 6, 10, 5, 15],
    [2, 11, 13, 4, 7, 8],
    [6, 3, 12, 14, 1, 9],
    [15, 13, 8, 0, 7, 2],
    [1, 14, 3, 7, 8, 12],
    [13, 9, 6, 15, 0, 2],
    [5, 6, 4, 11, 9, 10],
    [14, 12, 1, 3, 7, 8],
    [2, 15, 0, 6, 13, 9],
    [4, 6, 5, 11, 9, 10],
    [1, 8, 3, 7, 12, 14],
    [8, 15, 0, 3, 7, 12],
    [2, 6, 10, 9, 5, 13],
    [1, 4, 14, 8, 7, 11],
    [0, 3, 8, 12, 15, 7],
    [6, 5, 2, 9, 13, 10],
    [7, 11, 1, 14, 4, 8],
    [2, 9, 13, 6, 15, 0],
    [1, 2, 5, 10, 14, 13],
    [15, 0, 9, 8, 6, 7],
    [9, 12, 4, 6, 11, 3],
    [14, 5, 10, 13, 1, 2],
    [9, 8, 15, 0, 6, 7],
    [3, 6, 11, 9, 12, 4],
    [14, 13, 10, 1, 2, 5],
    [2, 12, 11, 13, 1, 6],
    [2, 8, 13, 4, 3, 15],
    [9, 5, 14, 0, 7, 10],
    [12, 11, 13, 6, 1, 2],
    [13, 2, 8, 4, 15, 3],
    [5, 7, 10, 9, 0, 14],
    [14, 0, 5, 12, 3, 11],
    [10, 1, 14, 5, 2, 13],
    [3, 15, 0, 12, 9, 6],
    [15, 0, 11, 4, 7, 8],
    [10, 13, 14, 2, 5, 1],
    [3, 15, 6, 9, 12, 0],
    [11, 0, 7, 4, 8, 15],
    [1, 14, 13, 10, 5, 2],
    [1, 4, 14, 11, 0, 15],
    [9, 13, 10, 5, 6, 2],
    [14, 12, 8, 3, 1, 7],
    [0, 11, 15, 1, 14, 4],
    [13, 5, 9, 2, 10, 6],
    [1, 3, 14, 7, 8, 12],
    [9, 13, 5, 2, 10, 6],
    [10, 12, 15, 5, 3, 0],
    [1, 11, 10, 4, 5, 14],
    [12, 6, 13, 1, 2, 11],
    [3, 8, 9, 6, 4, 15],
    [6, 1, 14, 7, 8, 9],
    [11, 13, 2, 12, 6, 1],
    [5, 10, 0, 12, 15, 3],
    [1, 11, 7, 10, 12, 4],
    [13, 7, 0, 2, 14, 9],
    [8, 5, 3, 12, 2, 15],
    [10, 1, 14, 4, 5, 11],
    [7, 13, 9, 0, 14, 2],
    [6, 1, 12, 8, 15, 3],
    [5, 4, 10, 1, 11, 14],
    [10, 12, 15, 4, 1, 3],
    [12, 7, 11, 0, 6, 9],
    [8, 13, 5, 3, 14, 2],
    [4, 10, 1, 3, 12, 15],
    [11, 6, 7, 0, 9, 12],
    [8, 13, 2, 3, 14, 5],
    [5, 7, 10, 0, 9, 14],
    [2, 14, 12, 3, 1, 13],
    [4, 9, 10, 0, 15, 7],
    [6, 5, 11, 12, 3, 8],
    [13, 12, 2, 14, 1, 3],
    [7, 15, 0, 4, 10, 9],
    [3, 5, 8, 12, 6, 11],
    [10, 4, 9, 15, 7, 0],
    [6, 5, 11, 10, 4, 9],
    [15, 8, 12, 3, 0, 7],
    [10, 13, 2, 1, 14, 5],
    [6, 9, 10, 11, 5, 4],
    [8, 15, 12, 3, 7, 0],
    [10, 1, 5, 2, 14, 13],
    [4, 8, 15, 11, 0, 7],
    [2, 9, 0, 7, 14, 13],
    [13, 7, 5, 8, 2, 10],
    [0, 3, 12, 15, 4, 11],
    [2, 14, 7, 9, 13, 0],
    [13, 10, 2, 8, 5, 7],
    [0, 11, 3, 4, 12, 15],
    [1, 6, 5, 10, 11, 12],
    [4, 9, 14, 6, 11, 1],
    [8, 12, 13, 2, 3, 7],
    [15, 10, 14, 0, 1, 5],
    [9, 4, 1, 6, 14, 11],
    [12, 13, 7, 2, 3, 8],
    [15, 1, 5, 0, 10, 14],
    [7, 8, 12, 9, 6, 3],
    [4, 10, 11, 2, 5, 13],
    [8, 6, 7, 9, 14, 1],
    [0, 12, 15, 7, 3, 8],
    [5, 13, 2, 11, 10, 4],
    [6, 9, 1, 7, 8, 14],
    [12, 15, 7, 0, 3, 8],
    [10, 5, 13, 4, 11, 2]
]

ALL_VALUES = [
    [1, 7, -5, -1, -7, 5],
    [-9, -7, -9, 1, 9, 1],
    [-2, 2, 8, -2, 2, -2],
    [-7, 4, 7, -9, 1, 7],
    [9, -3, -7, -5, -5, 9],
    [7, -2, 7, -4, -4, -4],
    [-9, 1, 2, -5, 2, 1],
    [5, -2, 2, 5, -9, -1],
    [5, 5, -7, 7, -1, -1],
    [-4, 2, -9, 2, -3, 6],
    [-9, 4, 9, -9, 4, 7],
    [-3, 3, -2, -1, -1, 4],
    [3, 1, -8, -8, 1, 6],
    [-2, 7, 4, -4, -2, -4],
    [8, -8, 2, -1, 5, -2],
    [-7, 3, 1, 7, -6, -6],
    [5, -9, 9, 8, -2, -1],
    [-9, 7, -7, 3, 3, 3],
    [-4, -1, -5, -5, 7, 6],
    [-1, -4, 5, 4, -2, -3],
    [1, -5, -4, 2, 2, 1],
    [1, -2, -9, -4, 2, 7],
    [-9, 2, 7, 6, 9, -9],
    [-2, 1, -4, -7, 1, 7],
    [-2, 8, -4, 9, -3, -3],
    [8, 8, -7, -3, -7, -8],
    [3, -8, -8, 1, 6, 3],
    [9, -2, -4, 9, -9, 9],
    [-8, -8, -4, 1, 8, 1],
    [4, 8, 4, -2, -4, -4],
    [2, -5, -5, 2, 4, 2],
    [-9, 9, 9, 8, -4, -1],
    [-3, 5, 9, -4, -5, -3],
    [-6, 3, 4, -8, 1, 2],
    [-6, 7, 7, -4, -2, -4],
    [-2, -9, 1, 2, -2, 9],
    [4, -3, -3, -1, 5, 8],
    [-9, -2, -8, 8, -5, 9],
    [-7, -9, -7, 9, 1, 1],
    [4, 3, 3, 3, -4, -8],
    [-7, 9, -3, -1, -1, 9],
    [-5, 5, -9, 1, 3, 5],
    [1, 6, -2, -3, 4, -6],
    [1, 5, -5, -5, -5, 1],
    [-1, 5, 8, 3, -6, -4],
    [-4, -1, -4, -1, 4, 6],
    [2, 7, -8, 4, 8, -9],
    [3, 1, 1, 1, -5, -9],
    [6, -4, -4, -3, 9, -4],
    [-2, -2, 5, -7, 3, 2],
    [-8, -7, 9, -9, 1, 1],
    [-1, -6, 9, -1, -6, 8],
    [4, 1, 6, -6, 1, -8],
    [9, -6, -9, 6, -1, 2],
    [8, 7, -1, -3, -8, 8],
    [2, -1, 1, -7, -2, 7],
    [3, 5, -5, -3, 1, -1],
    [6, -9, -2, -1, 9, 6],
    [4, -9, 9, 1, -9, -7],
    [4, 1, -7, 5, 5, -8],
    [-2, -4, 5, -3, 7, -3],
    [-7, -7, 1, 3, 2, 6],
    [-3, -1, 5, -4, -1, 8],
    [-9, -3, 8, 2, -3, 5],
    [4, -9, 9, -4, 1, -8],
    [9, 8, -3, 8, -1, -9],
    [-8, 8, -5, -1, -5, 9],
    [6, -9, 1, -9, 2, 5],
    [-2, -6, -1, 5, 6, 6],
    [-4, -7, 9, 4, -1, 2],
    [5, -5, 1, 9, -9, -1],
    [9, -8, -9, 5, -8, 1],
    [7, -7, 5, 9, -1, -7],
    [-4, 6, 4, -4, 2, -1],
    [-9, -7, -9, 9, 3, 1],
    [7, 6, 6, -9, 1, -8],
    [-6, -1, -3, 8, -4, 8],
    [2, -8, -2, -2, 7, 2],
    [-8, -1, 8, 8, -1, 4],
    [6, -6, 1, -9, 1, -2],
    [8, -4, -6, -1, -6, 8],
    [8, 2, 2, 3, -8, -9],
    [-9, 8, -9, 3, 2, 5],
    [-1, -1, -3, 9, 5, -1],
    [1, 3, -3, -8, -5, 9],
    [9, 7, 5, -3, -1, -9],
    [-9, -9, 1, -8, 9, 1],
    [9, 9, -8, -7, -6, -1],
    [-5, 2, 3, 5, 1, -6],
    [-6, -8, 3, 4, 3, 4],
    [-5, -5, 8, 8, -1, -1],
    [1, -6, -2, 2, 4, -1],
    [9, -4, -1, 8, 9, -9],
    [1, -7, -9, 9, -1, 4],
    [7, -4, -2, -4, -6, 6],
    [-9, -9, 3, 8, 3, 2],
    [7, -1, -1, 9, -1, -4],
    [7, -7, 1, 1, 7, -9],
    [9, 7, 9, -9, -2, -1],
    [1, 9, 3, -9, 2, -7],
    [4, -1, -4, -2, -2, 4],
    [1, 2, -5, -8, -7, 8],
    [9, -2, -9, 8, 4, -1],
    [-8, 6, 1, 1, -6, 4],
    [-1, -2, 9, -4, -4, 9],
    [6, 2, 3, -7, -1, -3],
    [-4, 2, 4, -4, -4, 1],
    [-1, 3, -6, 4, 9, -4],
    [7, 7, -2, -1, -7, 6],
    [-8, -8, -7, 2, 8, 5],
    [2, 2, -7, 2, -6, 6],
    [-1, 8, -3, -6, -4, 8],
    [-1, 1, 5, -5, -1, 2],
    [-8, 8, -8, -4, -3, 1],
    [-7, 5, -7, 3, 3, 1],
    [-4, -2, 4, -1, -3, 6],
    [-8, 2, -8, 7, 5, 8],
    [-8, 1, 1, 6, -6, 3],
    [-1, -5, -1, 7, 5, -1],
    [-2, 1, -1, 4, 2, -6],
    [-1, -4, 3, 6, -4, 3],
    [-7, 9, 2, -5, 1, -9],
    [-2, -3, 3, -3, 8, 1],
    [5, 1, 5, -7, -5, -5],
    [9, -3, 6, -1, -4, -4],
    [-6, 1, 3, 4, -4, 2],
    [1, 5, -8, -3, -5, 9],
    [8, 3, -3, -8, 4, 8],
    [8, -5, -8, -2, -4, 8],
    [2, 5, -7, 2, -8, 4],
    [-7, -3, 8, -8, -3, 8],
    [-2, 5, -1, -2, 9, -5],
    [3, -8, 1, 6, 1, -8],
    [3, -8, -3, 7, -2, 4],
    [-7, 7, -9, 1, -7, 3],
    [-2, 9, -9, 9, -3, 8],
    [8, 4, -9, 4, 3, -9],
    [6, 9, -6, -2, -3, -4],
    [-2, -4, 9, -4, -9, 9],
    [3, -9, 1, 1, 4, -8],
    [-9, -1, 4, -5, 3, 8],
    [1, 9, 2, -9, -9, -6],
    [-4, -3, 2, 8, 3, -3],
    [-7, -9, 4, 1, 6, 5],
    [9, -8, -2, -3, -4, 9],
    [5, 1, 5, 2, -8, -5],
    [-3, -4, -2, -1, 7, 9],
    [8, -8, 5, -4, -2, 2],
    [3, 1, 8, -7, -7, -7],
    [-6, -1, 6, -2, 6, 6],
    [-7, -2, -6, 9, 7, -1],
    [3, 4, 3, 1, -5, -9],
    [6, -9, 9, -9, 9, 2],
    [-2, -2, 7, -3, 9, -1],
    [-2, -6, 6, 2, 9, -9],
    [-3, -1, 9, -9, 7, 8],
    [9, 1, -9, 3, -9, -7],
    [-1, -7, -4, 7, 9, -6],
    [9, 4, 4, 4, -9, -9],
    [-7, -1, -4, 7, 9, -4],
    [2, -9, -8, 1, 4, 2],
    [-8, 2, -1, -3, 2, 8],
    [-7, -2, 9, -1, 7, 6],
    [7, 3, -6, 2, -6, -6],
    [-3, -9, -3, 9, -4, 9],
    [4, 7, -7, 1, 4, -7],
    [6, -4, -4, -3, 9, -4],
    [-8, 2, -8, 1, 2, 5],
    [2, -2, 9, -2, -9, 6],
    [1, 3, -8, -7, -2, 2],
    [-2, 8, -3, 7, -8, 8],
    [-6, 7, -1, -4, 7, -3],
    [3, 1, 3, -5, 5, -6],
    [-2, -1, 9, -9, -6, 9],
    [5, 1, 2, -8, -8, 2],
    [1, -3, 9, -9, -5, 3],
    [8, -2, 6, -5, 9, -8],
    [-8, 5, 1, -9, 9, -9],
    [-2, -4, -2, 7, 5, -1],
    [7, 4, -7, -9, 4, 1],
    [9, -5, -5, 5, -3, -2],
    [1, 2, -7, -5, 2, 5],
    [-3, 1, 2, -6, 6, -2],
    [-1, -3, 2, 3, -7, 9],
    [-9, 1, -9, -7, 2, 9],
    [-7, -3, -1, -7, 9, 9],
    [-6, 5, 2, -5, 2, 2],
    [-6, 2, 5, 2, 3, -6],
    [8, -3, 9, -5, -1, -2],
    [-2, -5, 2, -9, 2, 9],
    [-8, 8, 7, -2, 9, -1],
    [4, -6, 4, 3, 1, -6],
    [-7, -6, -1, 9, 9, -5],
    [9, -2, 7, -3, -6, -7],
    [1, 4, -8, 4, -6, 3],
    [-1, -4, 7, -7, 9, 5],
    [-2, 2, -9, 2, -1, 5],
    [-2, 2, 1, 8, 8, -8],
    [-2, -4, 8, -8, 8, -5],
    [3, 4, -6, 2, 1, -7],
    [5, -1, 7, -7, -5, 1],
    [6, -1, 5, -1, -3, -1],
    [9, -9, -9, 1, -2, 2],
    [7, 9, -1, -9, 1, -7],
    [-1, 8, 8, 9, -9, -1],
    [-8, -9, -6, 1, 9, 2],
    [-3, -5, -5, -5, 9, 7],
    [-6, 1, 6, 6, -8, 2],
    [-8, -9, 1, 6, 4, 1],
    [-2, 9, -2, -1, 8, -4],
    [2, 2, 4, -2, -1, -5],
    [-3, 9, -1, -9, 8, 7],
    [8, -8, -7, 2, 3, -6],
    [2, -9, 2, 6, -4, 4],
    [8, -4, -5, 8, -2, -7],
    [-4, -1, 6, 9, -3, -1],
    [1, -9, 6, 4, 4, -6],
    [-7, 9, 1, -3, -5, 3],
    [6, 8, -4, 4, -8, -1],
    [-9, 8, -8, 2, -8, 4],
    [-8, -9, 2, 8, 4, 3],
    [-5, -1, -1, -2, 7, 7],
    [-8, 4, -8, -3, 1, 8],
    [6, -9, 9, 7, -2, -2],
    [-2, -2, 3, 3, -8, 6],
    [-3, -1, 9, 6, -9, 9],
    [-2, 1, -8, 3, -3, 1],
    [-2, 9, -3, 3, 3, -8],
    [4, 6, -5, -2, -1, -2],
    [7, -4, -2, 1, -7, 4],
    [1, 7, 1, 4, -7, -9],
    [-8, 8, -2, 2, -5, 5],
    [4, -9, -1, -3, 9, 9],
    [2, -6, -6, 1, -9, 6],
    [7, -4, 7, -2, -5, -7],
    [3, -9, 5, -7, 5, 3],
    [-2, 5, -1, -3, 8, -1],
    [4, -6, -9, 3, 1, 6],
    [2, -2, -7, -1, 7, 4],
    [-7, 8, -7, 1, -5, 3],
    [-2, 9, -2, -9, 9, 8],
    [9, 4, -1, -9, -4, 2],
    [8, -8, 8, -2, -2, -3],
    [-6, -6, 1, 1, 4, 1],
    [-1, -9, 8, -3, 9, 8],
    [-9, -7, -1, 9, 4, 2],
    [7, -4, -8, -1, 8, 6],
    [-6, -6, -5, 6, -1, 8],
    [-7, 3, 3, 5, 1, -5],
    [-1, 7, -2, 9, -1, -1],
    [-7, 3, 2, 1, 3, -4],
    [-2, -1, 2, -7, 2, 1],
    [9, 4, -3, -2, -8, 2],
    [1, 2, -6, -6, -6, 6],
    [8, -8, 8, -2, -1, 5],
    [-9, 2, -7, -8, 9, 4],
    [-2, 5, -4, 9, -9, 9],
    [-5, -5, 9, -5, -3, 9],
    [4, 1, -6, 1, -4, 2],
    [2, -2, -8, 8, -3, 4],
    [8, -8, 5, -1, -1, 8],
    [1, 2, 5, -6, -6, -4],
    [6, -6, -1, -3, 7, -2],
    [3, 1, -9, 3, -7, 5],
    [-9, 7, 9, 6, 2, -9],
    [5, -1, 6, -3, -1, -3],
    [-7, -1, 9, 5, 3, -9],
    [-6, 1, -9, 1, -9, 9],
    [-9, 9, -2, -3, 6, 6],
    [-8, -2, -2, 9, -1, 8],
    [4, -8, 1, 5, 5, -8],
    [4, -9, 2, -5, 4, 2],
    [5, -2, -2, 9, -2, -4],
    [-9, -2, 7, -4, 4, 4],
    [8, 7, 8, -8, -1, -1],
    [-6, 7, 1, -5, -6, 2],
    [-9, -3, 9, -3, -5, 9],
    [-8, -8, 8, 4, 2, 3],
    [-8, 3, -6, 6, 1, 2],
    [-2, -2, -3, 7, 9, -1],
    [-2, -7, -3, 1, 8, 3],
    [2, -2, -9, 2, 1, -8],
    [9, 7, -1, 6, -9, -1],
    [-6, 6, 3, 4, -8, 1],
    [-6, -1, 8, 9, -6, -6],
    [8, -3, -1, 8, -8, -3],
    [1, 1, 6, -6, -9, 2],
    [9, -3, 1, -9, -4, 3],
    [9, -3, 7, -2, 7, -7],
    [-5, -5, 7, 2, -5, 1],
    [-5, 1, -8, 6, 8, -2],
    [-3, -3, 7, -5, -1, 5],
    [-9, 2, -5, 2, 3, 1],
    [9, 2, -9, -1, -5, 9],
    [7, -7, 5, 1, -5, -1],
    [6, -1, -2, -8, 6, 9],
    [1, 3, -9, 3, -7, 5],
    [-1, -3, 7, 7, -3, -5],
    [-3, -7, -7, 7, 8, -2],
    [1, -3, -7, 1, 1, 1],
    [7, 4, 1, -4, -1, -4],
    [-9, 2, -6, 2, -2, 9],
    [-9, 7, -2, -4, 9, 9],
    [1, -6, 9, 2, -9, -5],
    [8, 8, 8, -2, -1, -9],
    [1, -8, 3, 8, 1, -8],
    [-2, -4, 9, -2, -1, 7],
    [6, -6, 2, -4, -6, 2],
    [8, -8, -2, 4, 2, -4],
    [6, -9, 1, -6, 1, -6],
    [-6, 6, -4, -4, -1, 9],
    [9, 5, -1, -1, -9, 9],
    [1, -8, 2, 7, -7, 5],
    [-5, 6, 9, -3, -2, -5],
    [-8, 1, -7, 2, 2, 3],
    [-6, 7, -2, 2, 2, -2],
    [-3, 3, -8, 1, 1, -3],
    [6, 6, 8, -9, -9, 2],
    [8, -1, 5, -1, -1, -5],
    [-3, 6, 6, -1, -6, -1],
    [-9, 3, 3, -8, 1, 3],
    [6, 5, 3, -1, -1, -7],
    [9, -3, 3, -8, -4, 2],
    [7, 1, -7, -4, 2, -7],
    [5, -5, -9, 8, 9, -1],
    [2, -9, -9, 9, -5, 1],
    [6, -5, -6, -1, 8, -4],
    [4, -2, -1, 4, -6, 6],
    [1, -8, 2, 5, 5, -8],
    [8, 2, -8, -5, -1, 5],
    [1, -9, 4, -4, 7, -2],
    [-1, 5, 8, -8, -1, 6],
    [1, -6, -6, -6, 9, 2],
    [9, -3, 8, -7, -1, -6],
    [-5, 1, 2, 1, 1, -8],
    [-2, 6, -3, -7, 7, 6],
    [-2, 8, -2, 2, -3, 2],
    [-9, 2, 1, 9, -9, -7],
    [-4, 3, 3, -4, 7, -2],
    [-7, 5, 1, -8, 6, 5],
    [9, -7, -4, -2, 9, -6],
    [-9, 2, 1, -9, 3, 7],
    [-1, -4, -1, 9, 6, -3],
    [-9, -3, -2, 7, 3, 4],
    [6, -1, 8, 5, -4, -8],
    [-9, 1, -9, 9, -2, 5],
    [-9, 5, -1, 8, -6, 9],
    [-5, -9, 4, -1, 3, 5],
    [1, 5, -8, 5, 1, -5],
    [-2, -3, -3, 7, 8, -1]
]