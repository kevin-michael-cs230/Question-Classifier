import torch
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import transformers
import numpy as np
from os import path
import pandas as pd
from transformers import AutoTokenizer, TFXLNetModel, TFAutoModel, TFXLNetForSequenceClassification
from constants import *

def get_tokenize_func():
    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased', model_max_length=MAX_LEN)
    def tk(df):
        out = tokenizer(df['passage'], df['question'], padding='max_length', truncation=True)
        return out['input_ids'], out['token_type_ids'], out['attention_mask']
  
    return tk

def tokenize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data['output_ids'], data['token_type_ids'], data['attention_mask'] = zip(*data.apply(get_tokenize_func(), axis=1))
    for col in ['output_ids', 'token_type_ids', 'attention_mask']:
        data[col] = data[col].apply(lambda cell: np.array(cell))
    return data

def get_features(df: pd.DataFrame) -> list:
    output_ids = np.stack(df['output_ids'].values)
    token_type_ids = np.stack(df['token_type_ids'].values)
    attention_mask = np.stack(df['attention_mask'].values)
    stackx_features = np.stack(df[STACKX_COLUMNS].values)
    return [output_ids, token_type_ids, attention_mask, stackx_features]

def get_labels(df: pd.DataFrame, label_name: str):
    # Extract numpy array, and reshape to be rank-2
    return np.reshape(df[label_name].values, (-1, 1))

def load_pickled_datasets():
    train_set = pd.read_pickle(path.join(PROCESSED_DIR, TRAIN_SET))
    dev_set = pd.read_pickle(path.join(PROCESSED_DIR, DEV_SET))
    test_set = pd.read_pickle(path.join(PROCESSED_DIR,TEST_SET))
    return train_set, dev_set, test_set

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Drop binary labels because they will be recreated later
    data = data.drop(columns=['comprehension binary', 'interest binary'])

    # Average the labels for given passage,question pairs 
    # (because some passage,question pairs were labled multiple times)
    data = data.groupby(['passage', 'question'], as_index=False).mean()

    # Remove all questions that are not understandable
    data = data.loc[(data['understandable'] == 1)]

    # Sort the data
    data = data.sort_values(by='passage')

    return data

def prepare_and_save(load_path: str, save_path: str):
    """
    Loads pandas dataset from load_path, preprocesses it,
    tokenizes it, and then saves it to save_path

    load_path: str
        path to csv file
    save_path:
        path to pickled pandas dataframe
    """
    data = pd.read_csv(load_path)
    data = preprocess_data(data)
    data = tokenize_dataframe(data)
    data.to_pickle(save_path)

def get_interest_dataset(data: pd.DataFrame) -> pd.DataFrame:
    
    # Remove intermediate values (hopefully will lead to strong associations)
    interest = data.loc[(data['interest'] >= 4) | (data['interest'] <= 2)]

    # Create binary lable
    interest['interest_binary'] = (interest['interest'] > 3).astype(int)

    return interest

def get_comprehension_dataset(data: pd.DataFrame) -> pd.DataFrame:
    
    # Remove intermediate values (hopefully will lead to strong associations)
    comprehension = data.loc[(data['comprehension'] >= 4) | (data['comprehension'] <= 2)]

    # Create binary lable
    comprehension['comprehension_binary'] = (comprehension['comprehension'] > 3).astype(int)

    return comprehension

def get_balanced_dataset(data: pd.DataFrame, label_col: str, random_state = None) -> pd.DataFrame:
    """
    Given a dataframe, return a subset that has a balanced number of 
    positive and negative labels

    data:
        the dataset
    label_col:
        the column of binary labels to balance
    random_state (optional):
        randomness seed
    """
    num_labels_per_group = min(data[label_col].value_counts().values)
    groups = data.groupby(label_col)
    if random_state:
        return groups.apply(lambda x: x.sample(num_labels_per_group, random_state = random_state))
    else:
        return groups.apply(lambda x: x.sample(num_labels_per_group))



