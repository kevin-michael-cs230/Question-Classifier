import torch
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import transformers
from transformers import TFXLNetForSequenceClassification

def create_model(max_len: int, num_sx_features: int) -> tf.keras.Model:
    encoder = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=1) #Input is tokenized strings, output is embeddings and logits (logits means pre-sigmoid label)
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name='token_type_ids')
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    stackx_features = layers.Input(shape=(num_sx_features,), dtype=tf.float32, name='stackx_features')

    logits = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    ).logits
    encoder_preds = layers.Dense(1, activation='sigmoid', name='encoder_preds')(logits)
    merged = layers.Concatenate()([encoder_preds, stackx_features])
    final_pred = layers.Dense(1, activation='sigmoid', name='final_prediction')(merged)

    model = tf.keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask, stackx_features],
            outputs=[final_pred], name='classifier'
        )

    return model

def create_model_no_stackx(max_len: int, num_sx_features: int) -> tf.keras.Model:
    encoder = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=1) #Input is tokenized strings, output is embeddings and logits (logits means pre-sigmoid label)
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name='token_type_ids')
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    stackx_features = layers.Input(shape=(num_sx_features,), dtype=tf.float32, name='stackx_features') #Leave it here so we don't have to reformat the dataset. But it doesn't connect to anything

    logits = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    ).logits
    encoder_preds = layers.Dense(1, activation='sigmoid', name='encoder_preds')(logits)

    model = tf.keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask, stackx_features],
            outputs=[encoder_preds], name='classifier'
        )

    return model