import torch
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import transformers
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFXLNetModel, TFAutoModel, TFXLNetLMHeadModel, TFXLNetForSequenceClassification


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


def generate_muliple_inputs(passages: list, questions: list, max_len: int) -> [np.array, np.array, np.array]:
    nentries = len(passages)
    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased', model_max_length=max_len, truncation=True)
    input_ids_array = np.zeros((nentries, max_len))
    token_type_ids_array = np.zeros((nentries, max_len))
    attention_mask_array = np.zeros((nentries, max_len))

    for i in range(nentries):
        ids = tokenizer(passages[i], questions[i], return_tensors='np', padding='max_length', truncation=True)
        input_ids_array[i] = ids['input_ids']
        token_type_ids_array[i] = ids['token_type_ids']
        attention_mask_array[i] = ids['attention_mask']

    return [input_ids_array, token_type_ids_array, attention_mask_array]


def get_data(path: str) -> pd.DataFrame:
    with open(path, 'r') as f:
        df = pd.read_csv(f, nrows=6246)
    return df


def train_model(data: pd.DataFrame, train_size: int, max_len: int) -> tf.keras.Model:
    passages = list(data["passage"])
    questions = list(data["question"])
    labels = list(data["comprehension binary"])
    
    # Create model
    mod3 = create_model(max_len)
    # Compile model
    mod3.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.BinaryCrossentropy())
    # Tokenize inputs
    inputs = generate_muliple_inputs(passages[:train_size], questions[:train_size], max_len) # save last 1000 examples
    # Fit model to inputs
    history = mod3.fit(inputs, np.array(labels[:train_size]), batch_size=1, epochs=2)

    return mod3

def test_model(model: tf.keras.Model, data: pd.DataFrame, train_size: int, test_size: int, max_len: int):
    passages = list(data["passage"])
    questions = list(data["question"])
    labels = list(data["comprehension binary"])
    
    # Test model on new inputs (first test_size examples after training examples)
    test_inputs = generate_muliple_inputs(passages[train_size:train_size+test_size], questions[train_size:train_size+test_size], max_len)
    test_outputs = np.rint(model.predict(test_inputs, batch_size=1))
    test_labels = np.array(labels[train_size:train_size+test_size]).reshape((test_size, 1))
    
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


def main():
    data = get_data('labels_with_stackx.csv')
    train_size = 32
    test_size = 8
    max_len = 1024
    model = train_model(data, train_size, max_len)
    test_model(model, data, train_size, test_size, max_len)

if __name__ == "__main__":
    main()
