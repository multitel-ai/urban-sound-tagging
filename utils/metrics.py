import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score

def auprc(y_true, y_scores):
    """ Compute AUPRC for 1 class
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auc (float): the Area Under the Recall Precision curve
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    recall = np.concatenate((np.array([1.0]), recall, np.array([0.0])))
    precision = np.concatenate((np.array([0.0]), precision, np.array([1.0])))
    return auc(recall, precision)

def binary_confusion_matrix(y_true, y_scores):
    TN, FP, FN, TP = confusion_matrix(y_true, y_scores).ravel()
    return TN, FP, FN, TP

def compute_macro_auprc(y_true, y_scores, return_auprc_per_class=False):
    """ Compute macro AUPRC
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auprc_macro (float): the macro AUPRC
    """
    _, num_classes = y_true.shape
    auprc_scores = [auprc(y_true[:,i],y_scores[:,i]) for i in range(num_classes)]
    # nanmean to ignore nan for borderline cases
    auprc_macro = np.nanmean(np.array(auprc_scores))
    if return_auprc_per_class:
        return auprc_scores, auprc_macro
    else:
        return auprc_macro

def compute_micro_auprc(y_true, y_scores):
    """ Compute micro AUPRC
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auprc_macro (float): the micro AUPRC
    """
    auprc_micro = auprc(y_true.flatten(), y_scores.flatten())
    return auprc_micro

def compute_micro_F1(y_true, y_scores):
    """ Compute micro F1 @ 0.5
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            micro_F1 (float): the micro F1
    """
    micro_F1 = f1_score(y_true.flatten(), np.around(y_scores).flatten())
    return micro_F1