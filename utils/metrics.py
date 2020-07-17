from sklearn import metrics


class Metrics():
    def __init__(self):
        self.string = "不知道写啥，随便写写"

    def accuracy(self, y_true, y_pred):
        y_true, y_pred = self.preprocess(y_true, y_pred)
        return metrics.accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred):
        y_true, y_pred = self.preprocess(y_true, y_pred)
        return metrics.precision_score(y_true, y_pred, average='macro', zero_division=True)

    def recall(self, y_true, y_pred):
        y_true, y_pred = self.preprocess(y_true, y_pred)
        return metrics.recall_score(y_true, y_pred, average='macro', zero_division=True)

    def f1_score(self, y_true, y_pred):
        y_true, y_pred = self.preprocess(y_true, y_pred)
        return metrics.f1_score(y_true, y_pred, average='weighted', zero_division=True)

    def confusion_matrix(self, y_true, y_pred):
        y_true, y_pred = self.preprocess(y_true, y_pred)
        return metrics.confusion_matrix(y_true, y_pred)

    def preprocess(self, y_true, y_pred):
        _, y_pred = y_pred.topk(1, 1, True)
        y_pred = y_pred.view(1, -1)[0]
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        return y_true, y_pred
