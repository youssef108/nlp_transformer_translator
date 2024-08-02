from encoder_decoder import Encoder,Decoder,EncoderBlock,DecoderBlock
from blocks import PositionalEncoding,InputEmbeddings,ProjectionLayer,MultiHeadAttention,FeedForward
import torch
from torch import nn

class Trasformer(nn.Module):
    def __init__(self , encoder: Encoder, decoder:Decoder, src_embedding:InputEmbeddings,tgt_embedding:InputEmbeddings,src_position:PositionalEncoding,tgt_position:PositionalEncoding,linear_layer:ProjectionLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embedding
        self.trg_embed=tgt_embedding
        self.src_pos=src_position
        self.trg_pos=tgt_position
        self.linear_layer=linear_layer

    def encode(self,x,src_mask):
        x=self.src_embed(x)
        x=self.src_pos(x)
        return self.encoder(x,src_mask)
    def decode(self,x,encoder_out,src_mask,tgt_mask):
        x=self.trg_embed(x)
        x=self.trg_pos(x)
        return self.decoder(x,encoder_out,src_mask,tgt_mask)
    def project(self,x):
        x=self.linear_layer(x)
        return x
    

def build_trasformer(src_vocab_size: int, tgt_vocab_size:int, src_seq_len:int,target_seq_len:int,d_model:int=512,N:int=6 ,h:int=8,dropout=0.1,d_ff:int=2048 ):    
    """
    src_vocab_size: is used for input embedding in nlp ( I think it will not be used in trajectory prediction)
    src_seq_len:input sequence length(obsereved traj)

    trg_seq_len: output sequence length(prediction length)
    d_model : embedding dimention
    N:number of encoder blocks
    h:number of heads in multiattention
    d_ff: feedforward dimention
    """
    src_embedding=InputEmbeddings(d_model=d_model,vocab_size=src_vocab_size)
    tgt_embedding=InputEmbeddings(d_model=d_model,vocab_size=tgt_vocab_size)


    src_position=PositionalEncoding(d_model=d_model,seq_len=src_seq_len,dropout=dropout)
    trg_position=PositionalEncoding(d_model=d_model,seq_len=target_seq_len,dropout=dropout)

    encoder_blocks=[]
    for _ in range(N):
        self_attention_block=MultiHeadAttention(dropout,h,d_model)
        feed_forward=FeedForward(dropout,d_model,d_ff)
        encoder_block=EncoderBlock(d_model,self_attention_block,feed_forward,dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks=[]

    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttention(dropout,h,d_model)
        decoder_cross_attention_block=MultiHeadAttention(dropout,h,d_model)       
        feed_forward=FeedForward(dropout,d_model,d_ff)
        decoder_block=DecoderBlock(d_model,decoder_self_attention_block,decoder_cross_attention_block,feed_forward,dropout)
        decoder_blocks.append(decoder_block)

    encoder= Encoder(nn.ModuleList(encoder_blocks),d_model=d_model)
    decoder=Decoder(nn.ModuleList(decoder_blocks),d_model=d_model)

    linear_layer=ProjectionLayer(d_model,tgt_vocab_size)

    transformer=Trasformer(encoder,decoder,src_embedding,tgt_embedding,src_position,trg_position,linear_layer)

    #initialize paramerts using xavier method
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    
    return transformer
