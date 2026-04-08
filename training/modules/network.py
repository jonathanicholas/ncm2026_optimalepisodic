import math
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    """
    A torch Categorical class with action masking.
    """

    def __init__(self, logits, mask):
        self.mask = mask

        self.mask_value = torch.tensor(
            torch.finfo(logits.dtype).min, dtype = logits.dtype
        )
        logits = torch.where(self.mask, logits, self.mask_value)
        super(CategoricalMasked, self).__init__(logits = logits)


    def entropy(self):
        if self.mask is None:
            return super().entropy()
        
        p_log_p = self.logits * self.probs

        # compute entropy with possible actions only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype = p_log_p.dtype, device = p_log_p.device),
        )

        return -torch.sum(p_log_p, axis = 1)
    

class FlattenExtractor(nn.Module):
    """
    A flatten feature extractor.
    """
    def forward(self, x):
        # keep the first dimension while flatten other dimensions
        return x.view(x.size(0), -1)


class ValueNet(nn.Module):
    """
    Value baseline network.
    """
    
    def __init__(self, input_size):
        super(ValueNet, self).__init__()
        self.fc_value = nn.Linear(input_size, 1)
    
    def forward(self, x):
        value = self.fc_value(x) # (batch_size, 1)

        return value


class ActionNet(nn.Module):
    """
    Action network.
    """

    def __init__(self, input_size, output_size):
        super(ActionNet, self).__init__()
        self.fc_action = nn.Linear(input_size, output_size)
    
    def forward(self, x, mask = None):
        self.logits = self.fc_action(x) # record logits for later analyses

        # no action masking
        if mask == None:
            dist = Categorical(logits = self.logits)
        
        # with action masking
        elif mask != None:
            dist = CategoricalMasked(logits = self.logits, mask = mask)
        
        policy = dist.probs # (batch_size, output_size)
        action = dist.sample() # (batch_size,)
        log_prob = dist.log_prob(action) # (batch_size,)
        entropy = dist.entropy() # (batch_size,)
        
        return action, policy, log_prob, entropy


class SharedGRURecurrentActorCriticPolicy(nn.Module):
    """
    GRU recurrent actor-critic policy with shared actor and critic.
    """

    def __init__(
            self,
            feature_size,
            action_size,
            hidden_size = 128,
            kappa_squared = 0.,
        ):
        super(SharedGRURecurrentActorCriticPolicy, self).__init__()

        # network parameters
        self.feature_size = feature_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.kappa_squared = kappa_squared

        # input feature extractor
        self.features_extractor = FlattenExtractor()
        
        # recurrent neural network
        self.gru = nn.GRUCell(feature_size, hidden_size)

        # policy and value net
        self.policy_net = ActionNet(hidden_size, action_size)
        self.value_net = ValueNet(hidden_size)


    def forward(self, obs, hidden = None, mask = None):
        """
        Forward the net.
        """

        # extract input features
        features = self.features_extractor(obs)

        # initialize hidden states
        if hidden is None:
            hidden = torch.zeros(features.size(0), self.gru.hidden_size, device = obs.device)
        
        # add noise
        std = hidden.std(dim = 1, keepdim = True) # (batch_size, 1)
        epsilon = torch.randn_like(hidden) # (batch_size, hidden_size)
        hidden = math.sqrt(1 - self.kappa_squared) * hidden + math.sqrt(self.kappa_squared) * std * epsilon
        
        # iterate one step
        hidden = self.gru(features, hidden)

        # compute action
        action, policy, log_prob, entropy = self.policy_net(hidden, mask)

        # compute value
        value = self.value_net(hidden)

        return action, policy, log_prob, entropy, value, hidden



if __name__ == '__main__':
    # testing

    feature_size = 60
    action_size = 3
    batch_size = 16


    net = SharedGRURecurrentActorCriticPolicy(
        feature_size = feature_size,
        action_size = action_size,
    )

    # generate random test input
    test_input = torch.randn((batch_size, feature_size))
    test_mask = torch.randint(0, 2, size = (batch_size, action_size), dtype = torch.bool)

    # forward pass through the network
    action, policy, log_prob, entropy, value, hidden = net(test_input, mask = test_mask)

    print('action:', action)
    print('policy:', policy)
    print('log prob:', log_prob)
    print('entropy:', entropy)
    print('value:', value)
    print('hidden:', hidden)
