import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, dataset, tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len):
        super().__init__()
        self.dataset=dataset
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.seq_len=seq_len

        self.sos_token=torch.tensor([tokenizer_src.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos_token=torch.tensor([tokenizer_src.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token=torch.tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_target_pair=self.dataset[index]
        src_text=src_target_pair['translation'][self.src_lang]
        tgt_text=src_target_pair['translation'][self.tgt_lang]
        #encode the tokens using their ids
        enc_input=self.tokenizer_src.encode(src_text).ids
        dec_input=self.tokenizer_tgt.encode(tgt_text).ids
        #3 calculate number of padding tokens
        enc_number_of_padding= self.seq_len - len(enc_input) -2
        dec_number_of_padding=self.seq_len-len(dec_input)-1

        if enc_number_of_padding<0 or dec_number_of_padding<0:
            raise ValueError('sentence is too long for the seq_len')
        
        encoder_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(enc_input, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_number_of_padding, dtype=torch.int64)
        ]
        )
        ##add only sos to the decoder input 
        decoder_input=torch.cat(
        [
            self.sos_token,
            torch.tensor(dec_input, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_number_of_padding, dtype=torch.int64)
        ]
        )

        ## add only eos to the label( the expected ouptput of decoder)

        label= torch.cat(
        [
            torch.tensor(dec_input, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_number_of_padding, dtype=torch.int64)
        ]
        )
        assert encoder_input.size(0)== self.seq_len
        assert decoder_input.size(0)== self.seq_len
        assert label.size(0)== self.seq_len
        return {
            "encoder_input": encoder_input,# (seq_len)
            "decoder_input":decoder_input,#(seq_len)
            ## make a mask for padding values of the decoder
            "encoder_mask":(encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            ## make a mask for padding for decpder input as well as mask words from seeing the words after it
            "decoder_mask":(decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            "label":label,
            "src_text":src_text,
            "tgt_text":tgt_text}

def casual_mask(size):
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int) ## returns only the upper triangle of a matrix
    ## returns the lower diagonal by equating to zero
    return mask==0
