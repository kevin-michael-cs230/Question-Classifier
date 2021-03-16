from tensorflow import keras
import tensorflow.keras.callbacks
import json
import matplotlib.pyplot as plt

"""
Adapted from: https://stackoverflow.com/questions/42392441/how-to-record-val-loss-and-loss-pre-batch-in-keras

Keras Callback class to store and plot per-batch training metrics
 - Metrics: loss, accuracy, recall, precision, auc
 - For each plot, you can specify the title, x-axis label, y-axis label, whether
   to plot data from the validation split or not, and whether to save the plot
 - Metrics can be saved to and reloaded from <model name>.json

Example usage:

    training_metrics = TrainMetrics('testmodel1', epochs=EPOCHS)
    model.fit(X_train, Y_train, ..., callbacks=[training_metrics])
    training_metrics.plot_loss()
    training_metrics.save_metrics_json()
    reloaded_metrics = TrainMetrics('testmodel1', load_json=True)
    reloaded_metrics.plot_loss(title='The same loss as before', save_image=True)

"""

class TrainMetrics(keras.callbacks.Callback):

################################ Public Interface ################################

    def __init__(self, model_name, epochs=2, load_json=False):
        self.model_name = f'{model_name}-train'
        self.epochs = epochs
        self.json_file = f'{self.model_name}.json'
        if load_json:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
                self.epochs = data['epochs']
                self.losses = data['losses']
                self.accuracy = data['accuracy']
                self.precision = data['precision']
                self.recall = data['recall']
                self.auc = data['auc']
                self.val_losses = data['val_losses']
                self.val_accuracy = data['val_accuracy']
                self.val_precision = data['val_precision']
                self.val_recall = data['val_recall']
                self.val_auc = data['val_auc']
            print(f'Loaded training metrics from {self.json_file}')

    def plot_loss(self, title=None, xaxis_label='Batch', yaxis_label='Loss', from_validation=False, save_image=False):
        if from_validation:
            if title == None:
                title = 'Validation Set - Loss Per Batch'
            self.plot_metric('val_losses', self.val_losses, title, xaxis_label, yaxis_label, save_image)
        else:
            if title == None:
                title = 'Loss Per Batch'
            self.plot_metric('losses', self.losses, title, xaxis_label, yaxis_label, save_image)

    def plot_accuracy(self, title=None, xaxis_label='Batch', yaxis_label='Accuracy', from_validation=False, save_image=False):
        if from_validation:
            if title == None:
                title = 'Validation Set - Accuracy Per Batch'
            self.plot_metric('val_accuracy', self.val_accuracy, title, xaxis_label, yaxis_label, save_image)
        else:
            if title == None:
                title = 'Accuracy Per Batch'
            self.plot_metric('accuracy', self.accuracy, title, xaxis_label, yaxis_label, save_image)

    def plot_precision(self, title=None, xaxis_label='Batch', yaxis_label='Precision', from_validation=False, save_image=False):
        if from_validation:
            if title == None:
                title = 'Validation Set - Precision Per Batch'
            self.plot_metric('val_precision', self.val_precision, title, xaxis_label, yaxis_label, save_image)
        else:
            if title == None:
                title = 'Precision Per Batch'
            self.plot_metric('precision', self.precision, title, xaxis_label, yaxis_label, save_image)

    def plot_recall(self, title=None, xaxis_label='Batch', yaxis_label='Recall', from_validation=False, save_image=False):
        if from_validation:
            if title == None:
                title = 'Validation Set - Recall Per Batch'
            self.plot_metric('val_recall', self.val_recall, title, xaxis_label, yaxis_label, save_image)
        else:
            if title == None:
                title = 'Recall Per Batch'
            self.plot_metric('recall', self.recall, title, xaxis_label, yaxis_label, save_image)

    def plot_auc(self, title=None, xaxis_label='Batch', yaxis_label='AUC', from_validation=False, save_image=False):
        if from_validation:
            if title == None:
                title = 'Validation Set - AUC Per Batch'
            self.plot_metric('val_auc', self.val_auc, title, xaxis_label, yaxis_label, save_image)
        else:
            if title == None:
                title = 'AUC Per Batch'
            self.plot_metric('auc', self.auc, title, xaxis_label, yaxis_label, save_image)

    def save_metrics_json(self):
        data = {
            'epochs': self.epochs,
            'losses': self.losses,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'auc': self.auc,
            'val_losses': self.val_losses,
            'val_accuracy': self.val_accuracy,
            'val_precision': self.val_precision,
            'val_recall': self.val_recall,
            'val_auc': self.val_auc
        }
        with open(self.json_file, 'w') as f:
            json.dump(data, f)
        print(f'Saved training metrics to {self.json_file}')

################################################################################


    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.auc = []
        self.val_losses = []
        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []
        self.val_auc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.precision.append(logs.get('precision'))
        self.recall.append(logs.get('recall'))
        self.auc.append(logs.get('auc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.val_precision.append(logs.get('val_precision'))
        self.val_recall.append(logs.get('val_recall'))
        self.val_auc.append(logs.get('val_auc'))

    def plot_metric(self, metric_name, metric, title, xaxis_label, yaxis_label, save_image):
        num_batches = len(metric)
        epochs = list(range(1, self.epochs + 1))
        batches = list(range(1, num_batches + 1))
        batches_per_epoch = num_batches // self.epochs

        epoch_batchnums = list(range(1, num_batches + 1, batches_per_epoch))
        epoch_labels = [f'Epoch {i}' for i in range(1, self.epochs + 1)]

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_axes([0, 0, 1, 1]) # main axes
        ax.plot(batches, metric)
        ax.set_title(title)
        ax.set_xlabel(xaxis_label)
        ax.set_ylabel(yaxis_label)
        ax.set_xticks(epoch_batchnums)
        ax.set_xticklabels(epoch_labels)

        # In case there are a lot of epochs, prevents labels from colliding
        if len(epoch_labels) > 5:
            divider = len(epoch_labels) // 5
            i = 0
            for label in ax.get_xticklabels():
                if i % divider != 0:
                    label.set_visible(False)
                i += 1
        if save_image:
            fig.savefig(f'{self.model_name}-{metric_name}.png')
        plt.show()
