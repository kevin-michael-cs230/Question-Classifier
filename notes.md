

# Notes on stackx labels

- Show that method that successfully overfit for encoder alone does not work when stackx included
- Stackx labels were not effective in helping the model fit the data
- To evaluate the usefuleness of the stackx labels, created a model that used only the stackx labels (no encoder)
- Unable to fit the model to the training data.
    - Tried 300 epochs and could't get accuracy above 50%
    - Loss is unstable even with very low learning rate.
        - Setup: Even with all these modifications
            - Adam with lr = 1e-8, batch sizes of 64
            - Batch normalization
            - An 8-node hidden layer
        - Loss swings significantly between epochs and no consistent trend.
        - See only_stackx-train.json
- Thus, decided to ignore the stackx labels during training.

# Notes on interest

## Attempt to overfit

### model_overfit

```python
TRAIN_RATIO = .6
DEV_RATIO = .2
LEARNING_RATE = lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-5, decay_steps=50, decay_rate=0.9, staircase=True)
BATCH_SIZE = 3
EPOCHS = 14
```

- With 14 epochs, maximum accuracy achieved on epoch 11 (75.77%), and final accuracy was 70.75%
- Loss decreased consistently for the first few batches, but level at at around batch 7 near 0.6
- Even though goal was overfit, it good acceptable levels of accuracy on dev set:
    - Accuracy: 69.18%
    - F1 Score: 67.36%
- Overall, it looks like we are underfitting



        








Value of 0.000001 and 2 epochs lead to 58% on training and 54.5% on dev. 
Further training did not lead to higher accuracy

Tried seeing of model had capacity to fit. Ran 6 epochs on 100 training examples
to see if I could at least get it to overfit. loss: 0.6451 - accuracy: 0.6222
Dev: loss: 0.6841 - accuracy: 0.5566

Ran 6 epochs on 200 training examples. loss: 0.6424 - accuracy: 0.6300
Dev: loss: 0.6874 - accuracy: 0.5524
Then another 6 epcohs on another 100 training examples. loss: 0.6233 - accuracy: 0.6200
Dev: loss: 0.7047 - accuracy: 0.5594
Then another 3 epochs on another 400 examples. loss: 0.6606 - accuracy: 0.5725
Dev: loss: 0.7007 - accuracy: 0.5650
Then another 3 epochs on another 500 examples. loss: 0.6486 - accuracy: 0.6300
Dev: loss: 0.6761 - accuracy: 0.5944
Then another epoch on entire training set. loss: 0.6507 - accuracy: 0.6247
Dev: loss: 0.6624 - accuracy: 0.6238
Then another epoch on entire training set. 0.6324 - accuracy: 0.6424
Dev: loss: 0.7089 - accuracy: 0.5986 <- We've reached the point of overfit

Ran 8 epochs on 100 examples with 
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-5,
    decay_steps=50,
    decay_rate=0.9,
    staircase=True
)
Successfully overfit the model

With exponential decay, model appears to start overfitting after epoch 2.
Holding back 60 examples from the test set for validation after each epoch seems to match dev set well.

After further inspection, it appears that the model just leans toward always predicing positive because 
there are more positive than negative examples. This suggests that there is not a meaningful correlation in the data.
For example, trained a model and got 54% accuracy on dev set. Number of positive and negative predictions were
Positives: 711 Negatives: 4