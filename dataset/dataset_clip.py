import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import random
import numpy as np
import yaml
import torchvision.transforms as T
import pandas as pd
import pytorch_lightning as pl
from transformers import XGLMTokenizer, XGLMModel
try:
    from dataset.utils import load_dataset_file, read_lmdb_folder
    from dataset.video_transform import ConsistentVideoTransforms
except:
    from utils import load_dataset_file, read_lmdb_folder
    from video_transform import ConsistentVideoTransforms

class PSP_Dataset(Dataset):

    def __init__(self, path, tokenizer, config, phase, max_words=128, resize=256, input_size=224):
        self.config = config
        self.max_words = max_words

        self.resize = resize
        self.input_size = input_size
        
        self.raw_data = pd.read_csv(path + f'{phase}.csv', delimiter=',')

        self.tokenizer = tokenizer

        self.lmdb_path = config['data']['lmdb_path']
        self.phase = phase
        self.max_length = config['data']['max_length']

        self.transforms = ConsistentVideoTransforms(mode=phase)
        self.segment_size = 16
        self.stride = 8
    def __len__(self):
        return len(self.raw_data)
        # return 100
    
    def __getitem__(self, index):
        # print(index
        sample = self.raw_data.iloc[index]
        file_name = sample['name']
        p_glosses = sample['pseudo_gloss']
        # p_glosses = sample['text']
        # file_name = sample['imgs_path']
        # try:
        encoding = self.tokenizer(p_glosses, truncation=False, add_special_tokens=False ,return_tensors='pt')['input_ids']
        if len(encoding.shape) == 0:
            encoding = encoding.unsqueeze(0) 
        
        img_sample = self.load_imgs(file_name)
        # img_sample = torch.randn(2,2)
        # print(img_sample.shape)
        return {
            'file_name': file_name,
            'video_clips': img_sample,
            'input_ids': encoding.squeeze(0) if len(encoding.shape) > 1 else encoding,
        }
        # return file_name, img_sample, encoding['input_ids'].squeeze()
    
    def load_imgs(self, file_name):
        
        phase, file_name = file_name.split('/')
        folder = os.path.join(self.lmdb_path, phase)
        # print(folder, file_name)
        images = read_lmdb_folder(folder, file_name)
        # print(len(images))
        images = images[::2]
        # print(len(images))
        # exit(0)
        # print(type(images[0]))
        len_imgs = len(images)
        
        if len_imgs > self.max_length:
            images = images[:self.max_length]
            len_imgs = len(images)
        
        batch_image = []
        for i,img in enumerate(images):
            # print(img.shape)
            img = np.transpose(img, (1, 2, 0))
            # img = np.transpose(img, (0, 1, 2))
            img = Image.fromarray(img)
            batch_image.append(img)
        
        transformed_frames = self.transforms(batch_image)
        num_frames = transformed_frames.shape[0]
        if num_frames < self.segment_size:
            last_frame = transformed_frames[-1:]  # Shape: (1, 3, 224, 224)
            repetitions = self.segment_size - num_frames
            repeated_frames = last_frame.repeat(repetitions, 1, 1, 1)  # Repeat along the first dimension
            transformed_frames = torch.cat((transformed_frames, repeated_frames), dim=0)
        num_frames = len(transformed_frames)
        sublists = self.extract_sublists(num_frames, self.segment_size, self.stride)
        clips = []
        for (start, end) in sublists:
            clip = transformed_frames[start : end+1]
            clips.append(clip)
        
        return torch.stack(clips).permute(0, 2, 1, 3, 4)
        # except:
        #     print(file_name)
        #     print(num_frames)
        #     print(sublists)
        #     exit(0)
            
            

    
    def extract_sublists(self, len_list, sublist_size=16, slide_size=8):
        """
        Extract sublists of a given size from a list of numbers, with a sliding window.
        
        Parameters:
        numbers (list): List of consecutive numbers
        sublist_size (int): Size of each sublist to extract
        slide_size (int): Number of elements to shift the window for each step
        
        Returns:
        list: List of extracted sublists
        """
        numbers = list(range(len_list))
        sublists = []
        for i in range(0, len_list, slide_size):
            sublist = numbers[i:i+sublist_size]
            if len(sublist) < sublist_size:
                extra_len = sublist_size - len(sublist)
                sliced = numbers[i-extra_len:i]
                sublist = sliced + sublist
                if (sublist[0], sublist[-1]) not in sublists: sublists.append((sublist[0], sublist[-1]))
            else:
                if (sublist[0], sublist[-1]) not in sublists: sublists.append((sublist[0], sublist[-1]))
        return sublists
    
    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'


def collate_fn(input_batch):
    # Add <bos> token to the beginning of each sequence    
    batch = [seq['input_ids'] for seq in input_batch]
    clips = [item['video_clips'] for item in input_batch]  # Extract clips from each video
    
    # Pad the sequences to the maximum length in the batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=1)
    # Create attention masks
    attention_mask = (padded_batch != 1).long()
    
    
    
    # Find the maximum number of clips in the batch
    max_clips = max([clip_set.shape[0] for clip_set in clips])
    
    # Pad each video's clips to have the same number of clips (max_clips)
    padded_clips = []
    masks = []
    for clip_set in clips:
        num_clips = clip_set.shape[0]
        pad_size = max_clips - num_clips
        # Pad with zeros along the time dimension
        padded_clip_set = torch.cat([clip_set, torch.zeros((pad_size, *clip_set.shape[1:]))], dim=0)
        mask = torch.cat([torch.ones(num_clips), torch.zeros(pad_size)])  # Create a mask
        padded_clips.append(padded_clip_set)
        masks.append(mask)
    
    # Stack padded clips and masks into a batch
    padded_clips = torch.stack(padded_clips, dim=0)  # Shape: (batch_size, max_num_clips, C, D, H, W)
    masks = torch.stack(masks, dim=0)  # Shape: (batch_size, max_num_clips)
    
    return {
        'input_ids': padded_batch,
        'attention_mask': attention_mask,
        'video_clips': padded_clips,
        'video_masks': masks
    }

class DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            root_text_path,
            tokenizer_path,
            data_config: dict|str,
            resize=256,
            input_size=224,
            batch_size=1, 
            num_workers=1,
            data_ver=0):
        super().__init__()
        self.text_train = root_text_path

        self.tokenizer_path = tokenizer_path

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
        self.tokenizer = XGLMTokenizer.from_pretrained(tokenizer_path)
        # Ensure the tokenizer has the necessary special tokens
        # special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str|None = None):
        if stage == 'fit' or stage is None:
            # tran and valdiation dataset
            
            self.train_dataset = PSP_Dataset(path=self.text_train, tokenizer=self.tokenizer, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='train')
            self.val_dataset = PSP_Dataset(path=self.text_train, tokenizer=self.tokenizer, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='dev')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn)


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
    tokenizer_path = "/home/sobhan/Documents/Code/xglm-1.7B"
    root_text_path = '/home/sobhan/Documents/Code/PSP/data/pseudo_gloss_'
    phase = 'dev'
    data_module = DataModule(
        root_text_path,
        tokenizer_path,
        data_config=config,
        batch_size=8,
        num_workers=1,
    )

    data_module.setup()
    # train_dataset = data_module.train_dataset
    # val_dataset = data_module.val_dataset
    # print(val_dataset[5289])
    # print(len(val_dataset[5289]['input_ids'].shape) == 0)
    # print(len(val_dataset))
    from transformers import XGLMForCausalLM
    # model = XGLMForCausalLM.from_pretrained(tokenizer_path).model.embed_tokens
    train_dataloader = data_module.val_dataloader()
    # print(dataloader)
    tokenizer = XGLMTokenizer.from_pretrained(tokenizer_path)
    # Example training loop
    for idx, batch in enumerate(train_dataloader):
        print(batch['input_ids'].shape)
        print(batch['video_clips'].shape)
        print(batch['video_masks'][-1])
        # print(batch['input_ids'][0])
        # print(model(batch['input_ids']).shape)
        # exit(0)
        # print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
        # print(batch['attention_mask'].shape)
        # print(batch['attention_mask'][0])
        # print(len(batch['list_of_frames']))
        # for video in batch['list_of_frames']:
        #     print(video.shape)
        # print('Successfully loaded batch {}'.format(idx))

