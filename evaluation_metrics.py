import json
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

"""
Class to store and retrieve per-batch evaluation metrics
 - Metrics: accuracy, confusion matrix, classification report (per class f1 scores,
   recall, precision)
 - Can specify to save confusion matrix image (and its title)
 - Metrics can be saved to and reloaded from <model name>.json

Example usage:

    dev_preds = model.predict(dev_features, batch_size=BATCH_SIZE)
    dev_preds = np.rint(dev_preds).astype(int)
    eval_dataset = {'feats': dev_features, 'labels': dev_labels, 'preds': dev_preds}
    eval_metrics = EvalMetrics('testmodel2', dataset=eval_dataset)
    eval_metrics.plot_confusion_matrix()
    eval_metrics.save_metrics_json()
    reloaded_eval_metrics = EvalMetrics('testmodel2', load_json=True)
    reloaded_eval_metrics.plot_confusion_matrix()

"""


class EvalMetrics:

################################ Public Interface ################################

    def __init__(self, model_name, dataset={}, load_json=False):
        self.model_name = f'{model_name}-eval'
        self.json_file = f'{self.model_name}.json'
        if load_json:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
                self.feats = [np.array(feat) for feat in data['feats']]
                self.preds = np.array(data['preds'])
                self.labels = np.array(data['labels'])
            print(f'Loaded evaluation metrics from {self.json_file}')
        else:
            self.feats = dataset['feats']
            self.preds = dataset['preds']
            self.labels = dataset['labels']

        self.cr = classification_report(self.labels, self.preds, output_dict=True)

    def get_accuracy(self):
        return self.cr['accuracy']

    def get_f1score(self, avg_type='weighted avg'):
        return self.cr[avg_type]['f1-score']

    def get_precision(self, avg_type='weighted avg'):
        return self.cr[avg_type]['precision']

    def get_recall(self, avg_type='weighted avg'):
        return self.cr[avg_type]['recall']

    def get_classification_report(self):
        """
        Returns a dictionary of metrics.
        For example:
        {
            '0': {
                'precision': 0.6,
                'recall': 0.07780979827089338,
                'f1-score': 0.1377551020408163,
                'support': 347
            },
            '1': {
                'precision': 0.4693200663349917,
                'recall': 0.9401993355481728,
                'f1-score': 0.6261061946902655,
                'support': 301
            },
            'accuracy': 0.4783950617283951,
            'macro avg': {
                'precision': 0.5346600331674958,
                'recall': 0.5090045669095331,
                'f1-score': 0.3819306483655409,
                'support': 648
            },
            'weighted avg': {
                'precision': 0.5392983641463465,
                'recall': 0.4783950617283951,
                'f1-score': 0.3645971990894031,
                'support': 648
            }
        }
        where '0' is the class representing no comprehension and '1' is the
        class representing comprehension.
        """
        return self.cr

    def print_classification_report(self):
        print(classification_report(self.labels, self.preds))

    def plot_confusion_matrix(self, title='Confusion Matrix', save_image=False):
        cm = confusion_matrix(self.labels, self.preds)
        df_cm = pd.DataFrame(cm, columns=['Predicted: 0', 'Predicted: 1'], index=['Actual: 0', 'Actual: 1'])
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}).set_title(title)
        if save_image:
            plt.savefig(f'{self.model_name}-cm.png')
        plt.show()

    def save_metrics_json(self):
        data = {
            'feats': [feat.tolist() for feat in self.feats],
            'preds': self.preds.tolist(),
            'labels': self.labels.tolist()
        }
        with open(self.json_file, 'w') as f:
            json.dump(data, f)
        print(f'Saved evaluation metrics to {self.json_file}')

################################################################################
