import torch
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import transformers
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from transformers import AutoTokenizer, TFXLNetModel, TFAutoModel, TFXLNetLMHeadModel, TFXLNetForSequenceClassification


TRAINING_BATCH_SIZE = 2
TRAINING_EPOCHS = 3

CSV_PATH = 'labels_with_stackx.csv'
MAX_LEN = 1024


def generate_muliple_inputs(passages: list, questions: list, max_len: int) -> [np.array, np.array, np.array]:
    nentries = len(passages)
    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased', model_max_length=max_len, truncation=True)
    input_ids_array = np.zeros((nentries, max_len), dtype=int)
    token_type_ids_array = np.zeros((nentries, max_len), dtype=int)
    attention_mask_array = np.zeros((nentries, max_len), dtype=int)

    for i in range(nentries):
        ids = tokenizer(passages[i], questions[i], return_tensors='np', padding='max_length', truncation=True)
        input_ids_array[i] = ids['input_ids']
        token_type_ids_array[i] = ids['token_type_ids']
        attention_mask_array[i] = ids['attention_mask']

    return (input_ids_array, token_type_ids_array, attention_mask_array)
    
def get_data(csv_path):
    return pd.read_csv(csv_path).head(200)


def create_model(max_len: int) -> tf.keras.Model:
    encoder = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=1) #Input is tokenized strings, output is embeddings and logits (logits means pre-sigmoid label)
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids') 
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name='token_type_ids')
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    
    logits = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    ).logits
    out = layers.Dense(1, activation='sigmoid', name='classifier')(logits)
    
    model = tf.keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=[out], name='test_mod'
        )
    
    return model

def train_model(train_features, train_labels, max_len: int) -> tf.keras.Model:
    
    # Create model
    mod3 = create_model(max_len)
    # Compile model
    mod3.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.BinaryCrossentropy())
    # Fit model to inputs

    history = mod3.fit(train_features, train_labels, batch_size=TRAINING_BATCH_SIZE, epochs=TRAINING_EPOCHS)

    return mod3

def test_model(model: tf.keras.Model, test_features, test_labels):
    
    test_outputs = np.rint(model.predict(test_features, batch_size=1))
    print(test_outputs.shape)
    print(test_labels.shape)
    exit()
    
    true_negatives, true_positives, false_negatives, false_positives = 0, 0, 0, 0
    for i in range(test_size):
        if test_outputs[i][0] == test_labels[i][0]:
            if test_outputs[i][0] == 0:
                true_negatives += 1
            else:
                true_positives += 1
        else:
            if test_outputs[i][0] == 0:
                false_negatives += 1
            else:
                false_positives += 1
    f1score = true_positives / (true_positives + (false_positives + false_negatives)/2)

    print(f"Number of test examples: {test_size}")
    print(f"False negatives: {false_negatives}")
    print(f"False positives: {false_positives}")
    print(f"True negatives: {true_negatives}")
    print(f"True positives: {true_positives}")
    print(f"F1 score: {f1score}")

def get_train_and_test(csv_path, max_len, train_size, test_size):
    data = get_data(csv_path)
    nentries = len(data["passage"])
    passages = list(data["passage"])
    questions = list(data["question"])
    labels = np.array(list(data["comprehension binary"]))
    labels = np.reshape(labels, (labels.shape[0], 1))

    # Tokenize the inputs
    input_ids_array, token_type_ids_array, attention_mask_array = generate_muliple_inputs(passages, questions, max_len)
    
    # Shuffle the inputs
    input_ids_array, token_type_ids_array, attention_mask_array, labels = shuffle(input_ids_array, token_type_ids_array, attention_mask_array, labels)

    # Get train and test. They're random due to shuffling the entire dataset
    train_features = (input_ids_array[0:train_size], token_type_ids_array[0:train_size], attention_mask_array[0:train_size])
    train_labels = labels[0:train_size]

    test_features = (input_ids_array[train_size:train_size+test_size], token_type_ids_array[train_size:train_size+test_size], attention_mask_array[train_size:train_size+test_size])
    test_labels = labels[train_size:train_size+test_size]

    return train_features, train_labels, test_features, test_labels


def main():
    train_size = 128
    test_size = 32
    max_len = MAX_LEN

    train_features, train_labels, test_features, test_labels = get_train_and_test(CSV_PATH, max_len, train_size, test_size)

    model = train_model(train_features, train_labels, max_len)
    test_model(model, test_features, test_labels)

if __name__ == "__main__":
    main()
