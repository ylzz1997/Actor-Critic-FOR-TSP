import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder,self).__init__()
        self.config = config
        
        # Data config
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  # input sequence length (number of cities)
        self.input_dimension = config.input_dimension  # dimension of a city (coordinates)

        # Network config
        self.input_embed = config.input_embed  # dimension of embedding space
        self.num_neurons = config.hidden_dim  # dimension of hidden states (LSTM cell)


        self.W_embed = torch.nn.Conv1d(self.input_dimension,self.input_embed,1,bias = False)
        self.bn = torch.nn.BatchNorm1d(self.input_embed,affine=False)
        self.LSTM = torch.nn.LSTM(input_size = self.input_embed,hidden_size =self.num_neurons,bias = False)
    

    # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
    def forward(self,input):
        embedded_input = self.W_embed(input.transpose(1,2))
        embedded_input = self.bn(embedded_input).transpose(1,2)
        encoder_output, encoder_state = self.LSTM(embedded_input.transpose(0,1))
        return encoder_output, encoder_state
    
