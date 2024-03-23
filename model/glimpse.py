import torch
import torch.nn as nn
import torch.nn.functional as F

class Glimpse(nn.Module):

    def __init__(self, config):
        super(Glimpse,self).__init__()
    
        self.W_ref_g = torch.nn.Conv1d(config.hidden_dim,config.hidden_dim,1,bias = False)
        self.W_q_g = torch.nn.Linear(config.hidden_dim,config.hidden_dim,bias = False)
        self.v_g = torch.nn.Parameter(torch.randn((config.hidden_dim),requires_grad=True))
        

    def forward(self,ref, query, mask = None, current_task=None):
        # Attending mechanism
        encoded_ref_g = self.W_ref_g(ref.transpose(1,2)).transpose(1,2)  # [Batch size, seq_length, n_hidden]
        encoded_query_g = self.W_q_g(query)[:,None,:]  # [Batch size, 1, n_hidden]
        
        scores_g = torch.sum(self.v_g * torch.tanh(encoded_ref_g + encoded_query_g),-1)  # [Batch size, seq_length]
        # Attend to current task and tasks to apply only (Apply mask)
        if mask!= None and current_task!=None:
            attention_g = F.softmax(scores_g - 100000000. * (mask - current_task),dim=-1) # [Batch size, seq_length]
        else:
            attention_g = F.softmax(scores_g ,dim=-1)
        # 1 glimpse = Linear combination of reference vectors (defines new query vector)

        glimpse = torch.mul(ref, attention_g[:,:,None])
        glimpse = torch.sum(glimpse, 1)

        return glimpse
    
    