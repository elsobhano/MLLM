import os
import json
import lmdb
import torch
import pickle
from tqdm import tqdm
from typing import List
from transformers import MBartModel, MBartTokenizer

class MBartFeatureExtractor:
    def __init__(self, model_name=None, batch_size=5):
        """
        Initialize MBart model and tokenizer
        
        :param model_name: Pretrained MBart model name (default: large multilingual model)
        :param batch_size: Number of texts to process in each batch
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        # Load the pretrained model and tokenizer
        self.tokenizer = MBartTokenizer.from_pretrained(model_name)
        self.model = MBartModel.from_pretrained(model_name).to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Set batch size
        self.batch_size = batch_size
    
    def extract_features(self, texts, max_length=512, pooling_strategy='mean'):
        """
        Extract [CLS] token features from input texts using batched processing
        
        :param texts: List of input texts or a single text string
        :param max_length: Maximum sequence length
        :return: Tensor of extracted [CLS] token features
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Initialize list to store features        
        # Process texts in batches
        # Tokenize the batch
        inputs = self.tokenizer(
            texts,
            return_tensors='pt', 
            padding=True,
        ).to(self.device)
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Forward pass to get model outputs
            outputs = self.model(**inputs)
            
            # Extract the [CLS] token (first token) from the last hidden state
            if pooling_strategy == 'cls':
                batch_features = outputs.last_hidden_state[:, 0, :]
                batch_features = batch_features.cpu()
            elif pooling_strategy == 'mean':
                batch_features = outputs.last_hidden_state.mean(dim=1)
                batch_features = batch_features.cpu()
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        # Concatenate all features
        return batch_features


# model_path = "/home/sobhan/Documents/Code/mbart-large-cc25" 
def process_json_files(input_dir: str, output_dir: str, model_name: str, batch_size: int = 5):
    """
    Process JSON files, extract features in batches, and save to LMDB
    
    :param input_dir: Directory containing input JSON files
    :param output_dir: Directory to save LMDB databases
    :param batch_size: Number of texts to process in each batch
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize feature extractor with specified batch size
    feature_extractor = MBartFeatureExtractor(model_name=model_name, batch_size=batch_size)
    
    # Get list of JSON files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Process each JSON file
    for json_filename in tqdm(json_files, desc="Processing JSON files"):
        # Construct full file paths
        json_path = os.path.join(input_dir, json_filename)
        lmdb_path = os.path.join(output_dir, json_filename.replace('.json', '.lmdb'))
        
        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Prepare texts to extract features from
        texts = data['refined']
        
        # Extract features in batches
        features = feature_extractor.extract_features(texts)
        
        # Estimate LMDB map size with some buffer
        estimated_map_size = features.shape[0] * features.shape[1] * 8 * 10
        
        # Create LMDB environment
        env = lmdb.open(lmdb_path, map_size=estimated_map_size)
        
        with env.begin(write=True) as txn:
            
            serialized_feature = pickle.dumps(features.numpy())
                
                # Store in LMDB with key as string
            txn.put("data".encode('ascii'), serialized_feature)
        
        # Close LMDB environment
        env.close()
    
    print(f"Processed {len(json_files)} JSON files and saved features to LMDB.")

def read_lmdb_folder(lmdb_path, folder_name=None):
    """
    Read images from a specific folder key in the LMDB database.

    Parameters:
    - lmdb_path (str): Path to the LMDB database.
    - folder_name (str): The key (folder name) to retrieve images from.
    """
    # print(list_all_keys(lmdb_path))
    if folder_name == None:
        lmdb_file = lmdb_path
    else:
        lmdb_file = os.path.join(lmdb_path, f"{folder_name}.lmdb")
    
    env = lmdb.open(lmdb_file, readonly=True)
    with env.begin() as txn:
        data = txn.get("data".encode('ascii'))

    # Deserialize the list of images
    data = pickle.loads(data)

    # Convert back from CHW to HWC format for visualization
    # images = [np.transpose(img, (1, 2, 0)) for img in images]

    return data

if __name__ == "__main__":
    input_directory = "/home/sobhan/Documents/Code/ShareGPT4Video/{stage}_output"
    output_directory = "/home/sobhan/Documents/Datasets/desc_feat/{stage}"
    model_path = "/home/sobhan/Documents/Code/mbart-large-cc25"
    
    # for stage in ["train", "dev", "test"]:
    #     process_json_files(input_directory.format(stage=stage), output_directory.format(stage=stage), model_name=model_path)
    data = read_lmdb_folder(lmdb_path="/home/sobhan/Documents/Datasets/desc_feat/dev/01April_2010_Thursday_heute-6697.lmdb")
    # print(data.shape)