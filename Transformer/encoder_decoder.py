import torch
import torch.nn as nn
import math
from blocks import MultiHeadAttention,FeedForward,LayerNormalization,InputEmbeddings,ResidualConnection

class EncoderBlock(nn.Module):
    def __init__(self,d_model:int,Multi_head_attention_block: MultiHeadAttention,feedforwardblock:FeedForward,dropout:float):
        super().__init__()
        self.multi_head_attention_block=Multi_head_attention_block
        self.feed_forward_block=feedforwardblock
        # self.dropout=nn.Dropout(dropout)
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout,d_model=d_model) for _ in range(2)])

    def forward(self,x,src_mask):
        """
        src_mask: used for masking the padding values of the input
        """
        x=self.residual_connections[0](x, lambda x : self.multi_head_attention_block(x,x,x,src_mask))
        x= self.residual_connections[1](x,self.feed_forward_block)
        return x
class Encoder(nn.Module):
    def __init__(self,layers: nn.ModuleList,d_model:int):
        """
        layers: are number of encoder blocks 
        """
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization(d_model=d_model)
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self,d_model:int,self_multi_head_attention_block: MultiHeadAttention,Cross_multi_head_attention_block: MultiHeadAttention,feedforwardblock:FeedForward,dropout:float):
        super().__init__()
        self.self_attention=self_multi_head_attention_block
        self.cross_attention=Cross_multi_head_attention_block
        self.feed_forward=feedforwardblock
        self.residual_connection=nn.ModuleList([ResidualConnection(dropout=dropout,d_model=d_model) for _ in range(3)])
    
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        """
        src_mask : mask of the encoder
        tdt_mask: mask of the decoder
        """
        x= self.residual_connection[0](x,lambda x: self.self_attention(x,x,x,tgt_mask))
        x=self.residual_connection[1](x, lambda x: self.cross_attention(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connection[2](x,self.feed_forward)

        return x

class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList,d_model:int):

        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization(d_model=d_model)
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)

        return self.norm(x)
    
    