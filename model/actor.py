import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder import Pointer_decoder
from model.critic import Critic
from model.encoder import Encoder
from model.glimpse import Glimpse

class Actor(nn.Module):

    def __init__(self, config):
        super(Actor,self).__init__()
        self.config = config

        # Data config
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  # input sequence length (number of tasks)
        self.input_dimension = config.input_dimension  # dimension of a city (coordinates)

        # Network config
        self.input_embed = config.input_embed  # dimension of embedding space
        self.num_neurons = config.hidden_dim  # dimension of hidden states (LSTM cell)
        # variables initializer

        # Training config (actor)
        self.global_step = 0  # global step
        self.lr1_start = config.lr1_start  # initial learning rate
        self.lr1_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr1_decay_step = config.lr1_decay_step  # learning rate decay step
        self.is_training = config.training_mode

        # Training config (critic)
        self.global_step2 = 0  # global step
        self.lr2_start = config.lr1_start  # initial learning rate
        self.lr2_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr2_decay_step = config.lr1_decay_step  # learning rate decay step
        
        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.batch_idx = 0
        
        #Build Model
        self.encoder = Encoder(config)
        self.ptr = Pointer_decoder(self.config)

    # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
    def forward(self,input):
        encoder_output, encoder_state = self.encoder(input)
        positions, log_softmax = self.ptr(encoder_output.transpose(0,1),encoder_state)
        return positions, log_softmax
    
    # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
    def get_reward(self,input,positions):
            # Reorder input % tour
            positions = positions[:,:,None].repeat(1,1,2)
            # Sort
            ordered_input_ = torch.gather(input, 1, positions)
            # [batch size, seq length + 1, features] to [features, seq length + 1, batch_size]
            ordered_input_ = ordered_input_.permute(2, 1, 0)

            # Ordered coordinates
            ordered_x_ = ordered_input_[0]  # [seq length + 1, batch_size]
            # [batch_size, seq length]        delta_x**2
            delta_x2 = torch.square(ordered_x_[1:] - ordered_x_[:-1]).transpose(1, 0)
            ordered_y_ = ordered_input_[1]  # [seq length + 1, batch_size]
            # [batch_size, seq length]        delta_y**2
            delta_y2 = torch.square(ordered_y_[1:] - ordered_y_[:-1]).transpose(1, 0)
            
            # [batch_size, seq length]
            # Get tour length (euclidean distance)
            inter_city_distances = torch.sqrt(delta_x2 + delta_y2)  # sqrt(delta_x**2 + delta_y**2)
            distances = torch.sum(inter_city_distances, 1)[:,None]  # [batch_size,1]

            reward = distances.float()
            return reward
    
    def get_everylong(self,input,positions):
            # Reorder input % tour
            positions = positions[:,:,None].repeat(1,1,2)
            # Sort
            ordered_input_ = torch.gather(input, 1, positions)
            # [batch size, seq length + 1, features] to [features, seq length + 1, batch_size]
            ordered_input_ = ordered_input_.permute(2, 1, 0)

            # Ordered coordinates
            ordered_x_ = ordered_input_[0]  # [seq length + 1, batch_size]
            # [batch_size, seq length]        delta_x**2
            delta_x2 = torch.square(ordered_x_[1:] - ordered_x_[:-1]).transpose(1, 0)
            ordered_y_ = ordered_input_[1]  # [seq length + 1, batch_size]
            # [batch_size, seq length]        delta_y**2
            delta_y2 = torch.square(ordered_y_[1:] - ordered_y_[:-1]).transpose(1, 0)
            
            # [batch_size, seq length]
            # Get tour length (euclidean distance)
            inter_city_distances = torch.sqrt(delta_x2 + delta_y2)  # sqrt(delta_x**2 + delta_y**2)
            return inter_city_distances
    
    def loss1(self,reward,predictions,log_softmax):
        with torch.no_grad():
            reward_baseline = reward - predictions  # [Batch size, 1]
        loss = torch.mean(reward_baseline * log_softmax)
        return loss
        
    def loss2(self,reward,predictions):
        loss = F.mse_loss(reward, predictions) 
        return loss
