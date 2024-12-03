import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
import yaml
import torchvision.transforms as T
import pandas as pd

import copy

import pytorch_lightning as pl
from transformers import XGLMTokenizer
try:
    from dataset.utils import load_dataset_file, read_lmdb_folder
    from dataset.video_transform import ConsistentVideoTransforms
except:
    from utils import load_dataset_file, read_lmdb_folder
    from video_transform import ConsistentVideoTransforms


class SQA_Dataset(Dataset):

    def __init__(self, path, tokenizer, config, phase, max_words=128, resize=256, input_size=224):
        self.config = config
        self.max_words = max_words

        self.resize = resize
        self.input_size = input_size
        
        self.raw_data = pd.read_csv(path, delimiter='|')
        self.tokenizer = tokenizer

        self.lmdb_path = config['data']['lmdb_path']
        self.phase = phase
        self.max_length = config['data']['max_length']
        self.list = [key for key,value in self.raw_data.items()]   

        self.transforms = ConsistentVideoTransforms(mode=phase)
    def __len__(self):
        return len(self.raw_data)
        # return 100
    
    def __getitem__(self, index):
        sample = self.raw_data.iloc[index]
        # print(sample['name'])
        file_name = sample['name']
        question = sample['question']
        answer = sample['answer']
        if answer[-1] == '.':
            answer = answer.replace('.',' .')
        else:
            answer = answer + ' .'
        # key = self.list[index]
        # file_name = sample['imgs_path']
        prompt = f"Based on this Sign Language video, please answer the following question in German.\nQuestion: {question}\nAnswer:\n"
        # print(index)
        # print(prompt)
        # print(answer)
        # exit(0)
        prompt_encoded = self.tokenizer(prompt, truncation=False, add_special_tokens=False ,return_tensors='pt')
        encoding = self.tokenizer(answer, truncation=False, add_special_tokens=False ,return_tensors='pt')

        img_sample = self.load_imgs(file_name)
        
        return {
            'file_name': file_name,
            'video': img_sample,
            'input_ids': encoding['input_ids'].squeeze() if len(encoding['input_ids'].shape) > 1 else encoding['input_ids'],
            'prompt_ids': prompt_encoded['input_ids'].squeeze(),
        }
    
    def load_imgs(self, file_name):
        folder = os.path.join(self.lmdb_path, self.phase)
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
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        # exit(0)
        # print(file_name)
        # print(len_imgs)
        # print(type(images[0]))
        transformed_frames = self.transforms(batch_image)
        
        return transformed_frames

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'
    

if __name__ == "__main__":
    import yaml
    config = {
        'data': {
            'lmdb_path': 'src/sqa/data/lmdb',
            'max_length': 300,
        }
    }
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(config)

    tokenizer_path = "/home/sobhan/Documents/Code/xglm-1.7B"
    root_text_path = '/home/sobhan/Documents/Code/LLaMA-Adapter/SQA-Lightning/src/sqa/data/labels'
    phase = 'train'
    path = '/home/sobhan/Documents/Code/Sign2GPT/data/clean-qa.csv'
    tokenizer = XGLMTokenizer.from_pretrained(tokenizer_path)
    dataset = SQA_Dataset(path=path, tokenizer=tokenizer, config=config, phase=phase)
    print(len(dataset))
    dataset[0]


