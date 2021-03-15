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
from typing import Tuple, List

def get_tokenize_func() -> object:
    """ A closure that returns a tokenization function that can be mapped to every
    row in dataframe

    Returns:
        object: [description]
    """

    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased', model_max_length=MAX_LEN)
    def tk(row: pd.DataFrame) -> Tuple[list, list, list]:
        """ Tokenize a single row of a pandas dataframe

        Args:
            row (pd.DataFrame): a single row of a dataframe

        Returns:
            Tuple[list, list, list]: 
                three columns: input_ids, token_type_ids, attention_mask
        """
        out = tokenizer(row['passage'], row['question'], padding='max_length', truncation=True)
        return out['input_ids'], out['token_type_ids'], out['attention_mask']
  
    return tk

def tokenize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """ Tokenize the passage and question of a 
    dataframe and append the tokenizations as new columns

    Args:
        data (pd.DataFrame): the dataframe containing passages and questions to be tokenized

    Returns:
        pd.DataFrame: the dataframe with three new columns (the tokenized data) appended.
        Each element in the new columns is a numpy array of length MAX_LENGTH
    """
    data['output_ids'], data['token_type_ids'], data['attention_mask'] = zip(*data.apply(get_tokenize_func(), axis=1))
    for col in ['output_ids', 'token_type_ids', 'attention_mask']:
        data[col] = data[col].apply(lambda cell: np.array(cell))
    return data

def get_features(df: pd.DataFrame) -> List[np.ndarray]:
    """ Given a tokenized dataframe, extract the features as a list of numpy arrays

    Args:
        df (pd.DataFrame): a dataframe that has already been tokenized

    Returns:
        List[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: a list of numpy arrays
        The shape of the first three arrays is (num_entries x MAX_LEN)
        The shape of the last array (stackx_features) is (num_entries x NUM_STACKX_FEATURES)
        where num_entries =  the number of entries in the dataset = the number of rows in df

    """
    output_ids = np.stack(df['output_ids'].values)
    token_type_ids = np.stack(df['token_type_ids'].values)
    attention_mask = np.stack(df['attention_mask'].values)
    stackx_features = np.stack(df[STACKX_COLUMNS].values)
    return [output_ids, token_type_ids, attention_mask, stackx_features]

def get_labels(df: pd.DataFrame, label_name: str) -> np.ndarray:
    """ Given a tokenized dataframe, extract the labels as numpy array

    Args:
        df (pd.DataFrame): a dataframe that has already been tokenized
        label_name (str): the name of the dataframe column that contains the labels

    Returns:
        np.ndarray: an array of shape (num_entries x 1)
        where num_entries =  the number of entries in the dataset = the number of rows in df

    """
    # Extract numpy array, and reshape to be rank-2
    return np.reshape(df[label_name].values, (-1, 1))


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """ Get average scores for passage, question pairs that have multiple
    labels, drop non-understandable questions and return the result sorted

    Args:
        data (pd.DataFrame): the dataframe to process

    Returns:
        pd.DataFrame: the processed dataframe
    """
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
    """ Remove entries with moderate values for interest
    and add a column for a binary interest label

    Args:
        data (pd.DataFrame):

    Returns:
        pd.DataFrame: 
    """
    
    # Remove intermediate values (hopefully will lead to strong associations)
    interest = data.loc[(data['interest'] >= 4) | (data['interest'] <= 2)]

    # Create binary label
    interest['interest_binary'] = (interest['interest'] > 3).astype(int)

    return interest

def get_comprehension_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """ Remove entries with moderate values for comprehension
    and add a column for a binary comprehension label

    Args:
        data (pd.DataFrame):

    Returns:
        pd.DataFrame: 
    """
    
    # Remove intermediate values (hopefully will lead to strong associations)
    comprehension = data.loc[(data['comprehension'] >= 4) | (data['comprehension'] <= 2)]

    # Create binary label
    comprehension['comprehension_binary'] = (comprehension['comprehension'] > 3).astype(int)

    return comprehension

def get_balanced_dataset(data: pd.DataFrame, label_col: str, random_state = None) -> pd.DataFrame:
    """
    Given a dataframe, return a subset that has a balanced number of 
    positive and negative labels. 

    data:
        the dataset
    label_col:
        the column of binary labels to balance
    random_state (optional):
        randomness seed
    """
    num_labels_per_group = min(data[label_col].value_counts().values)
    groups = data.groupby(label_col, as_index=False)
    if random_state:
        return groups.apply(lambda x: x.sample(num_labels_per_group, random_state = random_state)).reset_index(drop=True)
    else:
        return groups.apply(lambda x: x.sample(num_labels_per_group)).reset_index(drop=True)

def kfold_split(features: List[np.ndarray], labels: np.ndarray, num_folds: int) -> list:
    """ Create a k-fold set of train and test folds

    Args:
        features List[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: the features
        labels (np.array): the labels
        num_folds (int): number of folds to generate

    Returns:
        list: a list of folds where each entry is a dictionary containing
            train_features
            train_labels
            test_features
            test_labels
        
    """


    token_ids = np.array_split(features[0], num_folds)
    token_type = np.array_split(features[1], num_folds)
    attention_mask = np.array_split(features[2], num_folds)
    stackx_features = np.array_split(features[3], num_folds)

    label_folds = np.array_split(labels, num_folds)

    folds = []
    for i in range(num_folds):
        
        train_token_ids = np.concatenate([token_ids[k] for k in range(num_folds) if k != i])
        train_token_type = np.concatenate([token_type[k] for k in range(num_folds) if k != i])
        train_attention_mask = np.concatenate([attention_mask[k] for k in range(num_folds) if k != i])
        train_stackx_features = np.concatenate([stackx_features[k] for k in range(num_folds) if k != i])
        train_features = [train_token_ids, train_token_type, train_attention_mask, train_stackx_features]
        
        test_token_ids = token_ids[i]
        test_token_type = token_type[i]
        test_attention_mask = attention_mask[i]
        test_stackx_features = stackx_features[i]
        test_features = [test_token_ids, test_token_type, test_attention_mask, test_stackx_features]

        train_labels = np.concatenate([label_folds[k] for k in range(num_folds) if k != i])
        test_labels = label_folds[i]
        folds.append({'train_features': train_features,
                    'test_features': test_features,
                    'train_labels': train_labels, 
                    'test_labels': test_labels})

    return folds

def get_balance(labels: np.ndarray) -> dict:
    """
    Get stats about the balance of positive and negative labels
    """
    unique, counts = np.unique(labels, return_counts=True)
    balance = dict(zip(unique, counts))
    balance['percent_pos'] = balance[1] / (balance[0] + balance[1])
    return balance

def get_subset(ftrs: list, start: int, stop: int) -> list:
    return [ftrs[0][start:stop], ftrs[1][start:stop], ftrs[2][start:stop], ftrs[3][start:stop]]

def split_datasets(data: pd.DataFrame, train_frac: float, dev_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a shuffled dataset and split fractions, split the dataset into train, dev and test sets 

    Args:
        data (pd.DataFrame): a shuffled dataset
        train_frac (float): the fraction of the input dataset to be training examples
        dev_frac (float): the fraction of the input dataset to be dev examples

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of (train_set, dev_set, test_set)
    """

    # Get train, test, dev splits
    num_entries = len(data)
    train_cutoff = int(num_entries * train_ratio)
    dev_cutoff = train_cutoff + int(num_entries * dev_ratio)

    train_set = data[:train_cutoff]
    dev_set = data[train_cutoff:dev_cutoff]
    test_set = data[dev_cutoff:]
    
    return train_set, dev_set, test_set






        

    




