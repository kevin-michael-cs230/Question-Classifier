import torch
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import transformers
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFXLNetModel, TFAutoModel, TFXLNetLMHeadModel, TFXLNetForSequenceClassification

SEED = 42
CSV_PATH = 'labels_with_stackx.csv'
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 0.0001

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


def train_model(data: pd.DataFrame, train_size: int, max_len: int) -> tf.keras.Model:
    passages = list(data["passage"])
    questions = list(data["question"])
    labels = list(data["comprehension binary"])

    # Create model
    mod3 = create_model(max_len)
    # Compile model
    mod3.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.BinaryCrossentropy())
    # Tokenize inputs
    inputs = generate_muliple_inputs(passages[:train_size], questions[:train_size], max_len) # tokenize the number of examples specified by train_size
    # Fit model to inputs
    history = mod3.fit(inputs, np.array(labels[:train_size]), batch_size=BATCH_SIZE, epochs=EPOCHS)

    return mod3


def evaluate_model(model: tf.keras.Model, data: pd.DataFrame, train_size: int, dev_size: int, max_len: int, label_category: str):
    passages = list(data["passage"])
    questions = list(data["question"])
    labels = list(data[label_category])

    # Create vector embeddings for inputs from training set and dev set
    train_inputs = generate_muliple_inputs(passages[:train_size], questions[:train_size], max_len)
    dev_inputs = generate_muliple_inputs(passages[train_size:train_size+dev_size], questions[train_size:train_size+dev_size], max_len)

    # Training set predictions and accuracy
    train_predictions = model.predict(train_inputs, batch_size=BATCH_SIZE)
    train_outputs = np.rint(train_predictions)
    train_labels = np.array(labels[:train_size]).reshape((train_size, 1))
    train_correct = int(np.sum(train_outputs == train_labels))

    # Dev set predictions and accuracy
    dev_predictions = model.predict(dev_inputs, batch_size=BATCH_SIZE)
    dev_outputs = np.rint(dev_predictions)
    dev_labels = np.array(labels[train_size:train_size+dev_size]).reshape((dev_size, 1))
    dev_correct = int(np.sum(dev_outputs == dev_labels))

    print("-"*50)
    print(f"Evaluating model on \"{label_category}\"...")
    print("-"*50)
    print(f"Train Accuracy:\t{train_correct}/{train_size}\t{(train_correct/train_size):.2f}%")
    print("-"*50)
    print(f"Dev Accuracy:\t{dev_correct}/{dev_size} \t{(dev_correct/dev_size):.2f}%")
    print("-"*50)

def get_data(path: str, max_len: int) -> pd.DataFrame:
    # Initial dataset has 6246 questions
    data = pd.read_csv(path)
    # Drop all rows with questions that are not understandable
    data.drop(data[data.understandable == 0].index, inplace=True)
    # All questions are understandable so we can remove the 'understandable' column
    data.drop(columns="understandable", inplace=True)
    # Remove rows whos passage has more than 'max_len' words
    data.drop(data[data.passage.map(lambda x: x.count(" ") + 1) > max_len].index, inplace=True)
    # Remove rows with a comprehension value of 3
    data.drop(data[data.comprehension == 3].index, inplace=True)
    # Remove rows with an interest value of 3
    data.drop(data[data.interest == 3].index, inplace=True)
    # Randomly shuffle the dataframe, randomness can be seeded by including random_seed=SEED
    data = data.sample(frac=1)

    return data


def main():
    # Dataset contains 3172 usable questions
    train_size = 2048 # number of training examples
    dev_size = 256 # number of testing examples
    max_len = 1024 # maximum number of words in a passage
    trait_to_evaluate = "comprehension binary"

    data = get_data(CSV_PATH, max_len)
    model = train_model(data, train_size, max_len)
    evaluate_model(model, data, train_size, dev_size, max_len, trait_to_evaluate)

if __name__ == "__main__":
    main()
