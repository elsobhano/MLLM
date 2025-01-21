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

from dataset.utils import load_dataset_file, read_lmdb_folder, data_augmentation

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

        sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        self.seq = va.Sequential([
            # va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
            # va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.2)),
            # va.RandomCrop(size=(256, 256)),
            sometimes(va.RandomTranslate(x=10, y=10)),

            # sometimes(Brightness(min=0.1, max=1.5)),
            # sometimes(Contrast(min=0.1, max=2.0)),

        ])
        import pickle
        emb_pkl_dir = 'data/processed_words.phx_pkl'
        with open(emb_pkl_dir, 'rb') as f:
            self.dict_processed_words = pickle.load(f)
    def __len__(self):
        return len(self.raw_data)
        # return 10
    
    def __getitem__(self, index):
        # print(index)
        key = self.list[index]
        sample = self.raw_data[key]
        name_sample = sample['name']
        tgt_sample = sample['text']
        list_of_pg = self.dict_processed_words['dict_sentence'][tgt_sample]
        pg_id = [self.dict_processed_words['dict_lem_to_id'][pg] for pg in list_of_pg]
        
        img_sample = self.load_imgs(name_sample)
        # print(img_sample.shape)
        return name_sample, img_sample, tgt_sample, pg_id
    
    def load_imgs(self, file_name):
        phase, file_name = file_name.split('/')
        folder = os.path.join(self.lmdb_path, phase)
        # print(folder, file_name)
        images = read_lmdb_folder(folder, file_name)
        len_imgs = len(images)
        
        if len_imgs > self.max_length:
            images = images[:self.max_length]
            len_imgs = len(images)
        
        imgs = torch.zeros(len_imgs,3, 224,224)
        crop_rect, resize = data_augmentation(resize=(256, 256), crop_size=224, is_train=(self.phase=='train'))
        data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])
        batch_image = []
        for i,img in enumerate(images):
            # print(img.shape)
            img = np.transpose(img, (1, 2, 0))
            # img = np.transpose(img, (0, 1, 2))
            img = Image.fromarray(img)
            batch_image.append(img)
        
        if self.phase == 'train':
            batch_image = self.seq(batch_image)
        
        for i, img in enumerate(batch_image):
            img = img.resize(resize)
            img = data_transform(img).unsqueeze(0)
            imgs[i,:,:,:] = img[:,:,crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]]

        return imgs

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'
    
    def collate_fn(self, batch):
        tgt_batch,img_tmp,src_length_batch,name_batch = [],[],[],[]
        pgs = []

        for name_sample, img_sample, tgt_sample, pg_list in batch:

            name_batch.append(name_sample)
            img_tmp.append(img_sample)
            tgt_batch.append(tgt_sample)
            pgs.append(pg_list)

        max_len = max([len(vid) for vid in img_tmp])
        mask = torch.zeros((len(img_tmp), max_len), dtype=torch.long)
        for i in range(len(img_tmp)):
            src_length_batch.append(len(img_tmp[i]))
            mask[i, : len(img_tmp[i])] = 1.0
        src_length_batch = torch.tensor(src_length_batch).long()
        
        img_batch = torch.cat(img_tmp,0)
        
        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(tgt_batch, return_tensors="pt", padding = True, max_length=self.max_words, truncation=True)

        src_input = {}
        src_input['input_ids'] = img_batch
        src_input['attention_mask'] = mask
        src_input['name_batch'] = name_batch
        src_input['src_length_batch'] = src_length_batch
        
        return src_input, tgt_input, pgs
    
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
    tokenizer_path = "/home/sobhan/Documents/Code/GFSLT-VLP/pretrain_models/MBart_trimmed"
    # qa_csv_path = '/home/sobhan/Documents/Code/Sign2GPT/data/clean-qa.csv'
    qa_csv_path = '/home/sobhan/Documents/Code/Sign2GPT/data/clean-qa.csv'
    root_text_path = '/home/sobhan/Documents/Code/LLaMA-Adapter/SQA-Lightning/src/sqa/data/labels'
    phase = 'train'
    data_module = DataModule(
        root_text_path,
        qa_csv_path,
        tokenizer_path,
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
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_path)
    # Example training loop
    for idx, batch in enumerate(train_dataloader):
        print(batch['targets'].shape)
        print(batch['targets'][0])
        print(tokenizer.decode(batch['targets'][0], skip_special_tokens=True))
        print(batch['atts_targets'].shape)
        print(batch['atts_targets'][0])
        print('Successfully loaded batch {}'.format(idx))

