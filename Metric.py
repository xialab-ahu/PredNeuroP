from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from numpy import interp
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn import metrics

def scores(y_true, y_pred, th=0.5):
    """
    :param y_true: 真实标签
    :param y_true: 预测概率
    :param th: 阈值
    :return: metric scores
    :rtype: np.array
    """
    y_predlabel = [(0. if item < th else 1.) for item in y_pred]
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_predlabel).flatten()
    SPE = tn * 1. / (tn + fp)
    MCC = metrics.matthews_corrcoef(y_true, y_predlabel)
    precision_prc, recall_prc, _ = precision_recall_curve(y_true, y_pred)
    PRC = auc(recall_prc, precision_prc)
    metric = np.array([metrics.recall_score(y_true, y_predlabel), SPE, metrics.precision_score(y_true, y_predlabel),
         metrics.f1_score(y_true, y_predlabel), MCC, metrics.accuracy_score(y_true, y_predlabel),
         metrics.roc_auc_score(y_true, y_pred), PRC], dtype=float)
    confusion = np.array([tn, fp, fn, tp], dtype=int)

    metric_columns = ['SE', 'SP', 'PRE', 'F1', 'MCC', 'ACC', 'AUC', 'AUPRC']
    print(metric_columns)
    return metric, confusion

def input_args():
    """
    Usage:
    python metric.py 
    """

    parser = argparse.ArgumentParser(usage="Usage Tip;",
                                     description = "PreNeuroP Feature Extraction")
    parser.add_argument("--file", "-f", required = True,
                        help = "input file(csv),column1 is label,column2 is probability")

    parser.add_argument("--out", "-o", required=True, help="output path and filename")
    return parser.parse_args()

if __name__ == '__main__':
    args = input_args()
    df = pd.read_csv(args.file,sep='\t')
    metric,confusion = scores(df.iloc[:,0],df.iloc[:,1])
    print(metric)
    print(confusion)