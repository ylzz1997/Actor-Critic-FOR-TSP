import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from model.glimpse import Glimpse

# RNN decoder指针网络
class Pointer_decoder(nn.Module):

    def __init__(self, config):
        super(Pointer_decoder,self).__init__()
    
        self.temperature = config.temperature  # temperature parameter
        self.C = config.C  # logit clip
        self.training_mode = config.training_mode
        
        # Decoder LSTM cell        
        self.cell = torch.nn.LSTMCell(input_size = config.hidden_dim,hidden_size = config.hidden_dim,bias = False)

        self.glimpse = Glimpse(config)

        self.W_ref = torch.nn.Conv1d(config.hidden_dim,config.hidden_dim,1,bias = False)
        self.W_q = torch.nn.Linear(config.hidden_dim,config.hidden_dim,bias = False)
        self.v = torch.nn.Parameter(torch.randn((config.hidden_dim),requires_grad=True))
        # Variables initializer


    def forward(self,encoder_output, decoder_initial_state):
        # decoder_initial_state: Tuple Tensor (c,h) of size [batch_size x cell.state_size]
        # decoder_first_input: Tensor [batch_size x cell.state_size]
        
        self.encoder_output = encoder_output  # Tensor [Batch size x time steps x cell.state_size] to attend to
        
        self.h = encoder_output.permute(1,0,2) # Tensor [time steps x Batch size x cell.state_size]

        batch_size = encoder_output.size(0)  # batch size
        self.seq_length = encoder_output.size(1) # sequence length
        n_hidden = encoder_output.size(2)  # num_neurons
        
        self.depot_position = (torch.ones(batch_size)*(self.seq_length-1)).long().to(encoder_output.device)

        self.log_softmax = []  # store log(p_theta(pi(t)|pi(<t),s)) for backprop
        self.positions = []  # store task list for reward

        # Start from depot
        self.positions.append(self.depot_position)
        
        # Keep track of current city
        self.current_task = F.one_hot(self.depot_position, self.seq_length)
        
        # Keep track of visited cities
        self.mask = F.one_hot(self.depot_position, self.seq_length)



        # Decoder initial input is depot (start)
        decoder_first_input = torch.gather(self.h.transpose(0,1),1,self.depot_position[:,None,None].repeat(1,1,self.h.size(-1)))[:,0,:]
        decoder_initial_state = (decoder_initial_state[0][0],decoder_initial_state[1][0])
        # Loop the decoding process and collect results
        s, i = decoder_initial_state, decoder_first_input
        for step in range(self.seq_length - 1):
            s, i = self.decode(s, i, step)

        # Return to depot
        self.positions.append(self.depot_position)

        # Stack visited indices
        self.positions = torch.stack(self.positions, axis=1)  # [Batch,seq_length+1]
        
        # Sum log_softmax over output steps
        self.log_softmax = torch.sum(torch.stack(self.log_softmax,dim=1),dim=1,keepdim=True).to(encoder_output.device)  # [Batch,seq_length-1]
        
        # Return stacked lists of visited_indices and log_softmax for backprop
        return self.positions, self.log_softmax

    # One pass of the decode mechanism
    def decode(self, prev_state, prev_input, timestep):

        # Run the cell on a combination of the previous input and state
        # print(prev_input.size(), prev_state.size())
        output,state = self.cell(prev_input, prev_state)
        state = (output,state)
        # Attention mechanism
        masked_scores = self.attention(self.encoder_output, output)

        # Multinomial distribution
        prob = Categorical(logits = masked_scores)

        # Sample from distribution
        position = prob.sample()
        self.positions.append(position)
        # Store log_prob for backprop
        self.log_softmax.append(prob.log_prob(position))

        # Update current city and mask
        self.current_city = F.one_hot(position, self.seq_length)
        self.mask = self.mask + self.current_city

        # Retrieve decoder's new input
        new_decoder_input = torch.gather(self.h.transpose(0,1),1,position[:,None,None].repeat(1,1,self.h.size(-1)))[:,0,:]
        #new_decoder_input = self.h[position][0]
        
        return state, new_decoder_input

    # From a query (decoder output) and a set of reference (encoder_output)
    # predict a distribution over next decoder input
    def attention(self, ref, query):
        glimpse = self.glimpse(ref,query,self.mask,self.current_task) + query

        # Pointing mechanism with 1 glimpse
        encoded_ref = self.W_ref(ref.transpose(1,2)).transpose(1,2)  # [Batch size, seq_length, n_hidden]
        encoded_query = self.W_q(glimpse)[:,None,:]  # [Batch size, 1, n_hidden]
        scores = torch.sum(self.v * torch.tanh(encoded_ref + encoded_query),-1)   # [Batch size, seq_length]
        if not self.training_mode:
            scores = scores / self.temperature  # control diversity of sampling (inference mode)
        scores = self.C * torch.tanh(scores)  # control entropy

        # Point to cities to visit only (Apply mask)
        masked_scores = scores - 100000000. * self.mask  # [Batch size, seq_length]

        return masked_scores

# import config

# pd = Pointer_decoder(config=config.get_config()[0])
# a = pd(torch.rand(1,12,128),torch.rand(1,128))
# q = a[0]
# q = q[:,:,None].repeat(1,1,2)
# inp = torch.rand(1,12,2)
# c = torch.gather(inp,1,q)
