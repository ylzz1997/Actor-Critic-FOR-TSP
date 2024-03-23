import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import Encoder
from model.glimpse import Glimpse

class Critic(nn.Module):

    def __init__(self, config):
        super(Critic,self).__init__()
        self.config = config

        # Data config
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  # input sequence length (number of cities)
        self.input_dimension = config.input_dimension  # dimension of a city (coordinates)

        # Network config
        self.input_embed = config.input_embed  # dimension of embedding space
        self.num_neurons = config.hidden_dim  # dimension of hidden states (LSTM cell)

        # Baseline setup
        self.init_baseline = self.max_length / 2.

        # Training config
        self.is_training = config.training_mode
        self.encoder = Encoder(config)

        self.glimpse = Glimpse(config)

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.num_neurons,self.num_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(self.num_neurons,1)
        )
        
        self.b1 = torch.nn.Parameter(torch.tensor(self.init_baseline), requires_grad=True)
        

    def forward(self, input):
        encoder_output,encoder_state = self.encoder(input)
        frame = encoder_state[0]
        encoder_output,frame = encoder_output.transpose(0,1),frame.transpose(0,1)[:,0,:]
        glimpse = self.glimpse(encoder_output,frame)
        predictions = self.dense(glimpse) + self.b1
        
        return predictions


# import config

# pd = Critic(config=config.get_config()[0])
# a = pd(torch.rand(1,12,2))