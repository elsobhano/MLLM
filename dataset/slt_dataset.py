import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import numpy as np
import yaml
import torchvision.transforms as T
from vidaug import augmentors as va
from torchvision import transforms

import pytorch_lightning as pl
from transformers import MBartTokenizer
try:
    from dataset.utils import load_dataset_file, read_lmdb_folder, data_augmentation
except:
    from utils import load_dataset_file, read_lmdb_folder, data_augmentation

import warnings

# Suppress a specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)

SI_IDX,PAD_IDX, UNK_IDX,BOS_IDX, EOS_IDX = 0 ,1 ,2 ,3 ,4
class S2T_Dataset(Dataset):

    def __init__(self, path, tokenizer, config, phase, max_words=128, resize=256, input_size=224):
        self.config = config
        self.max_words = max_words

        self.resize = resize
        self.input_size = input_size
        
        self.raw_data = load_dataset_file(path)

        self.tokenizer = tokenizer

        self.lmdb_path = config['data']['lmdb_path']
        self.phase = phase
        self.max_length = config['data']['max_length']
        self.list = [key for key,value in self.raw_data.items()]   

    def __len__(self):
        return len(self.raw_data)
        # return 10
    
    def __getitem__(self, index):
        # print(index)
        key = self.list[index]
        sample = self.raw_data[key]
        name_sample = sample['name']
        tgt_sample = sample['text']
        
        img_sample = self.load_data(name_sample)
        # print(img_sample.shape)
        # print('Hello World!')
        return name_sample, img_sample, tgt_sample
    
    def load_data(self, file_name):
        phase, file_name = file_name.split('/')
        folder = os.path.join(self.lmdb_path, phase)
        # print(folder, file_name)
        data = torch.from_numpy(read_lmdb_folder(folder, file_name + '_desc'))
        return data

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'
    
    def collate_fn(self, batch):
        tgt_batch,img_tmp,src_length_batch,name_batch = [],[],[],[]

        for name_sample, img_sample, tgt_sample in batch:

            name_batch.append(name_sample)

            img_tmp.append(img_sample)

            tgt_batch.append(tgt_sample)

        max_len = max([len(vid) for vid in img_tmp])
        video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 16 for vid in img_tmp])
        left_pad = 8
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len = max_len + left_pad + right_pad
        padded_video = [torch.cat(
            (
                vid[0][None].expand(left_pad, -1),
                vid,
                vid[-1][None].expand(max_len - len(vid) - left_pad, -1),
            )
            , dim=0)
            for vid in img_tmp]
        
        img_tmp = [padded_video[i][0:video_length[i],:] for i in range(len(padded_video))]
        
        for i in range(len(img_tmp)):
            src_length_batch.append(len(img_tmp[i]))
        src_length_batch = torch.tensor(src_length_batch)
        
        img_batch = torch.cat(img_tmp,0)

        new_src_lengths = (((src_length_batch-5+1) / 2)-5+1)/2
        new_src_lengths = new_src_lengths.long()
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX,batch_first=True)
        img_padding_mask = (mask_gen != PAD_IDX).long()
        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(tgt_batch, return_tensors="pt", padding = True, max_length=self.max_words, truncation=True)

        src_input = {}
        src_input['input_ids'] = img_batch
        src_input['attention_mask'] = img_padding_mask
        src_input['name_batch'] = name_batch

        src_input['src_length_batch'] = src_length_batch
        src_input['new_src_length_batch'] = new_src_lengths
        
        return src_input, tgt_input
    
class DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            root_text_path,
            qa_csv_path,
            tokenizer,
            data_config: dict|str,
            resize=256,
            input_size=224,
            batch_size=1, 
            num_workers=1,
            data_ver=0):
        super().__init__()
        self.text_train = root_text_path + '.train'
        self.text_val = root_text_path + '.dev'
        self.text_test = root_text_path + '.test'

        self.qa_csv_path = qa_csv_path

        if type(data_config) == str:
            with open(data_config, 'r') as file:
                self.data_config = yaml.safe_load(file)
        else:
            self.data_config = data_config
        
        if data_ver != 0:
            self.data_config['data']['lmdb_path'] = self.data_config['data']['lmdb_path'] + f'-{data_ver}'

        self.resize = resize
        self.input_size = input_size

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        ####################Intialize Tokenizer####################
        self.tokenizer = tokenizer
        # Ensure the tokenizer has the necessary special tokens
        # special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str|None = None):
        if stage == 'fit' or stage is None:
            # tran and valdiation dataset
            
            self.train_dataset = S2T_Dataset(path=self.text_train, tokenizer=self.tokenizer, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='train')

            self.val_dataset = S2T_Dataset(path=self.text_val, tokenizer=self.tokenizer, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='dev')
            # self.val_dataset = self.train_dataset
        if stage == 'test' or stage is None:
            # test dataset
            self.test_dataset = S2T_Dataset(path=self.text_test, tokenizer=self.tokenizer, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.val_dataset.collate_fn)

    def test_dataloader(self):        
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.test_dataset.collate_fn)


if __name__ == "__main__":
    import yaml
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    
    # config = {
    #     'data': {
    #         'lmdb_path': '/home/sobhan/Documents/Code/LLaMA-Adapter/SQA-Lightning/src/sqa/data/lmdb',
    #         'max_length': 300,
    #     }
    # }
    # print(config)
    tokenizer_path = "/home/sobhan/Documents/Code/MLLM/MBart_trimmed"
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_path)
    # qa_csv_path = '/home/sobhan/Documents/Code/Sign2GPT/data/clean-qa.csv'
    qa_csv_path = '/home/sobhan/Documents/Code/Sign2GPT/data/clean-qa.csv'
    root_text_path = '/home/sobhan/Documents/Code/LLaMA-Adapter/SQA-Lightning/src/sqa/data/labels'
    phase = 'train'
    data_module = DataModule(
        root_text_path,
        qa_csv_path,
        tokenizer,
        data_config=config,
        batch_size=2,
        num_workers=1,
    )

    data_module.setup()
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    test_dataset = data_module.test_dataset

    # print(train_dataset[42]['input_ids'])
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    train_dataloader = data_module.train_dataloader()
    # print(dataloader)
    # Example training loop
    for idx, (src_input, tgt_input) in enumerate(train_dataloader):
        print(tgt_input['input_ids'].shape)
        print(tgt_input['attention_mask'].shape)
        print(tokenizer.batch_decode(tgt_input['input_ids'], skip_special_tokens=True))
        print(src_input['input_ids'].shape)
        print(src_input['attention_mask'].shape)
        print(src_input['src_length_batch'].shape)
        print('Successfully loaded batch {}'.format(idx))

