from datetime import datetime

# import matplotlib.pyplot as plt
import numpy as np
import visdom
from matplotlib import rcParams
from sklearn.metrics import (auc, confusion_matrix,
                             multilabel_confusion_matrix, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.utils.multiclass import unique_labels

rcParams.update({'figure.autolayout': True, 'figure.figsize': (6, 9)})


class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            self.time_training_started = datetime.now()
        self.env_name = str(self.time_training_started.strftime("%d-%m %Hh%M"))
        self.vis = visdom.Visdom(env=self.env_name)

        # windows
        self.loss_win = None
        self.loss_epoch_win = None
        self.current_batch_win = None
        self.initial_time_win = None
        self.confusion_matrix_win = None
        self.roc_score_win = None

        # common attributes
        self.epoch_current = 0
        self.epoch_total = 0
        self.operation = None

    def plot_loss(self, loss, step, name):
        self.loss_win = self.vis.line(
            [loss], [step],
            win=self.loss_win,
            name=name,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (média dos últimos 10 lotes)',
            ))

    def plot_epoch_loss(self, epoch_history):
        '''Espera uma lista, onde cada elemento é uma lista

        Params:
            epoch_history: cada
        '''

        # mean_per_epoch = [(i, np.mean(loss_history[-10:])) for i, loss_history in enumerate(epoch_history)]
        mean_per_epoch = [
            np.mean(loss_history[-10:]) for loss_history in epoch_history
        ]

        self.loss_epoch_win = self.vis.line(
            mean_per_epoch, [x for x in range(len(mean_per_epoch))],
            win=self.loss_epoch_win,
            update='replace' if self.loss_epoch_win else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch Loss(média dos últimos 10 lotes do Epoch)',
            ))

    def plot_roc_auc_score(self, y_true, y_pred, epoch):
        """FIXME! briefly describe function

        :param y_true:
        :param y_pred:
        :param epoch: int - current epoch
        :returns:
        :rtype:

        """

        lb = LabelBinarizer()
        lb.fit(y_true)

        y_true_binarizado = lb.transform(y_true)
        y_pred_binarizado = lb.transform(y_pred)
        current_score = roc_auc_score(y_true_binarizado,
                                      y_pred_binarizado,
                                      average="macro")

        self.roc_score_win = self.vis.line(
            np.array([current_score]),
            [epoch],
            win=self.roc_score_win,
            update='append' if self.roc_score_win else None,
            opts=dict(xlabel='Epoch', ylabel='AUC', title='ROC Score'))

    def plot_confusion_matrix(self, y_true, y_pred):
        import matplotlib.pyplot as plt
        ax, fig = plot_confusion_matrix(y_true,
                                        y_pred,
                                        ['Alemão', 'Inglês', 'Espanhol'],
                                        normalize=True)
        # cm = multilabel_confusion_matrix(y_true, y_pred)
        # fig.fig(figsize=(6, 9))

        self.confusion_matrix_win = self.vis.matplot(
            fig, win=self.confusion_matrix_win)

    def plot_current_batch(self, current_batch, batch_size, dataset_length):
        self.current_batch_win = self.vis.text(
            F"Mode: {self.operation}<br><br>"
            F"Epoch: {self.epoch_current} of {self.epoch_total}<br>"
            F"Current Batch: {current_batch} of {dataset_length // batch_size}",
            win=self.current_batch_win,
            append=False)

    def update_elapsed_time(self, isFinished=False):
        current_elapsed_time_inSeconds = (
            datetime.now() - self.time_training_started).total_seconds()

        hours, remainder = divmod(current_elapsed_time_inSeconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        self.initial_time_win = self.vis.text(
            F"Time started: {self.env_name}<br>"
            F"Elapsed Time: {hours:2.0f}h{minutes:2.0f}m{seconds:2.0f}s<br>",
            win=self.initial_time_win,
            append=False)

        if isFinished:
            self.initial_time_win = self.vis.text(F"Training Finished.",
                                                  win=self.initial_time_win,
                                                  append=True)


def plot_confusion_matrix(y_true,
                          y_pred,
                          classes,
                          normalize=False,
                          title=None,
                          cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt

    # default
    cmap = plt.cm.Blues

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
    return ax, fig
