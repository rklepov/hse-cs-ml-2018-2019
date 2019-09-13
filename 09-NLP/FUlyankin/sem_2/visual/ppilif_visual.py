#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np     # Нумпай для векторов
import pandas as pd    # Пандас для табличек
# Округлять в табличках значения до второго знака
pd.set_option('precision', 2)

# Пакеты для графииков
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')   # Правильный стиль графиков

from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, roc_curve, auc, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, jaccard_similarity_score
import itertools

def visualize_coefficients(classifier, feature_names, n_top_features=10):
    ### визулизирует влияние факторов на целевую переменную

    coef = classifier.coef_.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])

    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 1 + 2 * n_top_features),
                feature_names[interesting_coefficients], rotation=60, ha="right");
    pass

def plot_precision_recall_curve_many(y_test, y_pred_probas, labels, figsize=(10, 10), title=''):
    plt.figure(figsize=figsize)
    f_scores = np.linspace(0.3, 0.9, num=7)
    lines = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.2f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    for i, y_pred_proba in enumerate(y_pred_probas):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.step(recall, precision, where='post', label=labels[i], lw=2)
        #plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    #plt.legend(prop={"size": 14}, loc='upper center', bbox_to_anchor=(0.5, -0.07))
    plt.legend(prop={"size": 14}, loc='lower left')
    plt.xlabel('Recall', size=15)
    plt.ylabel('Precision', size=15)
    plt.title(title, size=18)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    #plt.grid(color='lightgray', linestyle='dashed')
    plt.show()


def classification_quality_report(model, X_train, X_test, y_train, y_test,
                                  principle = 'maxf', cm_normalize=False):
    """
    Функция строит огромный отчёт с картинками и метрикам для бинарной классификации
        model - обученная модель
        X_train, X_test, y_train, y_test - выборка
        principle='maxf' - способ выбора порога
            'maxf' - максимальная f-мера
            'recall_0.5' - recall >= 0.5, а precision максимальный (можно указать любое число)
            'precision_0.9' - precision >= 0.9, а recall максимальный (можно указать любое число)
            действительное число - значение порога, например 0.3
        cm_normalize=False - нужно ли в confution_matrix пронормировать клетки к процентам

    """

    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.subplot2grid((12, 12), (0, 0), colspan=6, rowspan=6)
    ax2 = plt.subplot2grid((12, 12), (0, 6), colspan=6, rowspan=6)
    ax3 = plt.subplot2grid((12, 12), (7, 0), colspan=4, rowspan=4)
    ax4 = plt.subplot2grid((12, 12), (7, 4) ,colspan=8, rowspan=4)


    ### Кусочек с roc_auc
    y_hat_train = model.predict_proba(X_train)[:,1]
    y_hat_test = model.predict_proba(X_test)[:,1]

    fpr_train, tpr_train, thresholds_roc_train = roc_curve(y_train, y_hat_train)
    fpr_test, tpr_test, thresholds_roc_test = roc_curve(y_test, y_hat_test)

    roc_auc_train = roc_auc_score(y_train, y_hat_train)
    roc_auc_test = roc_auc_score(y_test, y_hat_test)

    ax1.plot(fpr_train, tpr_train, label='Train ROC AUC {0}'.format(roc_auc_train))
    ax1.plot(fpr_test, tpr_test, label='Test ROC AUC {0}'.format(roc_auc_test))
    ax1.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    # ax1.set_xlabel('False Positive Rate')
    # ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC curve', size=16)


    # Кусочек с precision recall curve на трэйне и тесте
    f_scores = np.linspace(0.4, 0.9, num=6)
    lines = []
    labels = []

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax2.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax2.annotate('f1={0:0.2f}'.format(f_score), xy=(0.9, y[45] + 0.02), size=9)


    precision_test, recall_test, trashholds_test  =  precision_recall_curve(y_test, y_hat_test)
    ax2.step(recall_test, precision_test, color='b', alpha=0.2, where='post')
    ax2.fill_between(recall_test, precision_test, step='post', alpha=0.2, color='b',
                     label = 'Test PR AUC: {0:0.2f}'.format(average_precision_score(y_test, y_hat_test)))
    # ax2.set_xlabel('Recall', size=15)
    # ax2.set_ylabel('Precision', size=15)
    ax2.set_title('Test Precision-Recall curve', size=16)
    ax2.set_xlim([-0.01, 1.00])
    ax2.set_ylim([-0.01, 1.01])


    # Подбор порога для классификации
    # максимальный precision при заданном recall
    if principle.split('_')[0] == 'recall':
        recall_cutoff = float(principle.split('_')[1])
        usl_1 = np.argwhere(recall_test >= recall_cutoff)
        usl_2 = np.argmax(precision_test[usl_1])
        cutoff = trashholds_test[usl_1 - 1][usl_2 - 1][0]
    # максимальный recall при заданном precision
    elif principle.split('_')[0] == 'precision':
        precision_cutoff = float(principle.split('_')[1])
        usl_1 = np.argwhere(precision_test >= precision_cutoff)
        usl_2 = np.argmax(recall_test[usl_1])
        cutoff = trashholds_test[usl_1 - 1][usl_2 - 1][0]
    # максимальная f-мера
    elif principle.split('_')[0] == 'maxf':
        f = 2*precision_test*recall_test/(precision_test + recall_test + 0.000001)
        position = np.argmax(f)
        cutoff = trashholds_test[position - 1]
    else:
        cutoff = float(principle)

    # рисую порог на roc_auc кривой
    position = np.argmin(np.abs(thresholds_roc_test - cutoff)) # нашли позицию ближайшего элемента к cutoff
    ax1.plot(fpr_test[position], tpr_test[position], 'o', markersize=10,
             fillstyle="none", c='k', mew=2, label="cutoff={0:.2f}".format(cutoff))
    ax1.legend(loc='lower right')

    # рисую порог на precision-recall кривой
    position = np.argmin(np.abs(trashholds_test - cutoff)) # нашли позицию ближайшего элемента к cutoff
    ax2.plot(recall_test[position], precision_test[position], 'o', markersize=10,
             fillstyle="none", c='k', mew=2, label="cutoff={0:.2f}".format(cutoff))
    ax2.legend(loc='lower right')


    # confusion matrix
    y_hat = y_hat_test > cutoff
    cm = confusion_matrix(y_test, y_hat)
    if cm_normalize:
        cm = np.round(100*cm.astype('float') / cm.sum() ,1)

    ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title('Confusion matrix', size=16)

    classes = ['0', '1']
    tick_marks = np.arange(len(classes))

    ax3.set_xticks(tick_marks, classes)
    ax3.set_yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax3.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=30)

    ax3.set_ylabel('True label', fontsize=16)
    ax3.set_xlabel('Predicted label', fontsize=16)


    # распределение прогнозов
    ax4.hist(y_hat_test)
    ax4.set_title('Test predict probability distribution', size=16)

    print('Порог:', cutoff)
    print('Принцип выбора порога:', principle)
    print('Процентов теста в бане: ', round(100*np.mean(y_hat_test > cutoff),2))
    print('lift: {0:.2f}'.format(len(y_test)*precision_score(y_test, y_hat_test > cutoff)/(sum(y_test))), '\n')

    report = classification_report(y_test, y_hat)
    print(report)
    pass
