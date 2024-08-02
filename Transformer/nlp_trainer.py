import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset,DataLoader,random_split
from pathlib import Path
from dataset import BilingualDataset,casual_mask
import torch.utils
from transformer import build_trasformer
from config import get_weights_file_path,get_config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx=tokenizer_tgt.token_to_id('[SOS]')
    eos_idx=tokenizer_tgt.token_to_id('[EOS]')
    #precompute encoder output only once
    encoder_ouput=model.encode(source,source_mask)
    decoder_input=torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) #(1 for batch, 1 for decoder input)
    while True:
        if decoder_input.size(1)==max_len:
            break
        decoder_mask=casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out=model.decode(decoder_input,encoder_ouput,source_mask,decoder_mask)
        proj=model.project(out[:,-1]) # get the projection of the most recent token only
        # select the token with max probability

        _,next_word=torch.max(proj,dim=1)

        decoder_input=torch.cat([decoder_input,torch.empty(1,1).type_as(source).fill_(next_word.item())],dim=1)

        if next_word==eos_idx:
            break
    return decoder_input.squeeze(0)# remove batch dimension




def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_state,writer,numexamples=2):
    model.eval()
    count=0

    # source_texts=[]
    # expected=[]
    # predicted=[]
    console_width=80

    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            encoder_input=batch['encoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)

            assert encoder_input.size(0)==1, "Batch size must be 1 for validation dataset"

            model_out=greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)


            source_text=batch['src_text'][0]

            target_text=batch['tgt_text'][0]
            #convert the predicted tokens to text
            model_out_text=tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)

            #print function provided by tqdm package
            print_msg('-'*console_width)
            print_msg(f'source: {source_text}')
            print_msg(f'target expected: {target_text}')
            print_msg(f'Predicted: {model_out_text}')

            if count==numexamples:
                break





def get_all_sentences(dataset,lang):
    for item in dataset:
        yield item['translation'][lang]


def build_tokenizer(config,dataset, lang):
    tokenizer_path=Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    dataset_raw=load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')

    tokenizer_src=build_tokenizer(config,dataset_raw,config['lang_src'])
    tokenizer_tgt=build_tokenizer(config,dataset_raw,config['lang_tgt'])

    train_size=int(0.9*len(dataset_raw))
    val_size=len(dataset_raw) -train_size
    train_dataset_raw, val_dataset_raw=random_split(dataset_raw,[train_size,val_size])
    train_dataset=BilingualDataset(train_dataset_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_dataset=BilingualDataset(val_dataset_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_len_src=0
    max_len_tgt=0

    for item in dataset_raw:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids=tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))

    print("Max length of source sentence ",max_len_src)
    print("max length of target sentence ",max_len_tgt)

    train_dataloader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=1,shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):
    model=build_trasformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model


def train_model(config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_dataset(config)

    model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)

    writer=SummaryWriter(config['experiment_name'])

    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)

    initial_epoch =0
    global_step=0
    if config['preload']:
        model_filename=get_weights_file_path(config,config['preload'])
        print(f'Preloading model {model_filename}')
        state=torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch=state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state['global_step']
    ## ignore_index is to ignore loss of padding tokens
    ## label smoothing is for the model to be less confident about its decisions = reduces overfitting
    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        batch_iterator=tqdm(train_dataloader,desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            model.train()
            encoder_input=batch['encoder_input'].to(device) # B,seq_len
            decoder_input=batch['decoder_input'].to(device) #B,seq_len
            encoder_mask=batch['encoder_mask'].to(device) # B,1,1,seq_len
            decoder_mask=batch['decoder_mask'].to(device) # B,1,seq_len,seq_len

            encoder_output=model.encode(encoder_input,encoder_mask) #B,seq_len_d_model
            decoder_ouput=model.decode(decoder_input,encoder_output,encoder_mask,decoder_mask)#B,seq_len,d_model
            proj_output=model.project(decoder_ouput) #B,seq_len,tgt_vocab_size

            label=batch['label'].to(device)#(B,seq_len)
            # (B*seq_len,tgt_vocab_size)
            loss=loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
            #update progress bar
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            #log the loss in tensor board

            writer.add_scalar('train loss',loss.item(),global_step)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            
            
            global_step+=1
        run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config['seq_len'],device,lambda x: batch_iterator.write(x),global_step,writer)
        model_filename=get_weights_file_path(config,f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),\
            'global_step':global_step,},
            model_filename)
        
if __name__ =='__main__':
    config=get_config()
    train_model(config)






