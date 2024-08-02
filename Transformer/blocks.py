import torch
import torch.nn as nn
import math
class InputEmbeddings(nn.Module):
    ### d_model is the dimetion of the embedding
    ## vocab_size is the total size of the number of words
    def __init__(self,vocab_size: int,d_model=512  ):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding =nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) ## the paper multiplies the embedding by the sqrt of the dimention
    
class PositionalEncoding(nn.Module):
    """ 

    seq_len: max seq length
    d_model: dimention of embedding
    
    """
    def __init__(self,seq_len,dropout:float,d_model=512):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        #create a matrix of shape [seq_len,d_model]
        pe=torch.zeros(seq_len,d_model)
        #create a vector of sape (seq_len,1)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) 
        Denomenator=torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))

        pe[:,0::2]= torch.sin(position*Denomenator)
        pe[:,1::2]=torch.cos(position*Denomenator)

        ## add a dimention for batch_size, (1,seq_len,d_model)
        pe=pe.unsqueeze(0)

        self.register_buffer("pe",pe)
    def forward(self,x):
        x=x+(self.pe[:, :x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self,d_model:int,epsilon :float=10**-6 ):
        super().__init__()
        self.epsilon=epsilon
        self.alpha=nn.Parameter(torch.ones(d_model))## multiplied
        self.beta=nn.Parameter(torch.ones(d_model))##added
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha *((x-mean)/(std+self.epsilon)) +self.beta
    
class FeedForward(nn.Module):
    def __init__(self,dropout,d_model:int =512,dff:int =2048):
        super().__init__()
        self.linear=nn.Sequential(nn.Linear(d_model,dff),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(dff,d_model))
    def forward(self,x):
        return self.linear(x)
class MultiHeadAttention(nn.Module):
    def __init__(self,dropout,h, d_model=512 ): 
        """
        h: number of heads in the multi head dimention, 
        the data is split in the embedding dimention. 
        """
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model % h==0 , "d_model is not divisble by h"
        #dimention of each head
        self.d_k=d_model//h
        ## matrices of parameters for Q,K and V
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        #matrix for output parameters
        self.w_o=nn.Linear(d_model,d_model)
        
        self.dropout=nn.Dropout(dropout)

        self.attention_scores=None
    @staticmethod
    def selfattention(query,key,value,dropout: nn.Dropout,mask=None):
        d_k=query.shape[-1]
        #(batch,h,seq_len,seq_len)
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9) ## hides interaction for specific words, for example padding values(filler words)
        attention_scores=attention_scores.softmax(dim=-1)
        if dropout:
            attention_scores=dropout(attention_scores)
        ## the attention_scores alone are used for visualization later
        return (attention_scores @ value), attention_scores 

    def forward(self,q,k,v,mask):
        query=self.w_q(q)# (batch,seq_len,d_model)
        key=self.w_k(k)
        value=self.w_v(v)
        # split the query by the embdeeding dimention , and make h the second dimention so the model can
        #see all of the sequence for each head (batch ,h,seq_len,d_k)
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores=self.selfattention(query,key,value,self.dropout,mask)
        #(batch,seq_len,h,d_k)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        #(batch,seq_len,d_model)
        # x=x.view(x.shape[0],-1,self.h*self.d_k)

        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self,dropout: float,d_model:int):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization(d_model=d_model)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class ProjectionLayer(nn.Module):
    """
    used to map the ouput embeddings of the decoder back to the vocabulary
    in tajectory prediction case hidden2normal can be used or linear layer from d_model to 2 (delta_x,delta_y)
    """
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
    

    
