# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:00:00 2022

@author: hey
"""

import numpy as np
import pandas as pd
import math
from numpy.linalg import  *
from six.moves import xrange
from catboost import Pool, CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MultiClassObjective(object):
    # approxes - indexed container of floats with predictions
    #            for each dimension of single object
    # target - contains a single expected value
    # weights - contains weight of the object
    # gamma - tunable focusing parameter in Focal Loss
    #
    # This function should return a tuple (der1, der2), where
    # - der1 is a list-like object of first derivatives of the loss function with respect
    # to the predicted value for each dimension.
    # - der2 is a matrix of second derivatives.
    #
    # approxes, target, weights are indexed containers of floats
    # (containers with only __len__ and __getitem__ defined).
    # weights parameter can be None.
    # Returns list of pairs (der1, der2)
    
    def __init__(self, gamma=2, class_w=None):
        self.class_w = class_w
        self.gamma = gamma
        
    def calc_ders_multi(self, approxes, target, weights):
        gamma = self.gamma
        class_w = self.class_w
        if class_w is None:
            class_w = np.ones(np.shape(approxes)[0])
        approxes = approxes - max(approxes)
        exp_approxes = np.exp(approxes)
        exp_sum = exp_approxes.sum()
        p = exp_approxes / exp_sum  # p：每一类的预测概率
        pt = p[int(target)]  # pt：目标类的预测概率
        grad = [] # 梯度向量
        hess = [] # hess矩阵
        for j in range(len(approxes)):
            # j != target
            pj = p[j] # pj: 偏导所在类预测概率
            temp = ((1-pt)**(gamma-1)) * gamma * pt * math.log(pt) - ((1-pt)**(gamma))
            der1 = -temp * pj
            if j == target:
                der1 = temp * (1-pt)
            hess_row = [] # 生成hess矩阵每一行
            for j2 in range(len(approxes)):
                pj2 = p[j2] # pj2: 二次偏导所在类预测概率
                if j == target:
                    # j == target, j2 != j
                    temp = (-(gamma**2) * pt * math.log(pt) + (gamma * math.log(pt) + 2 * gamma + 1) * (1 - pt))
                    der2 = -pt * pj2 * ((1 - pt)**(gamma - 1)) * temp
                    # j == target, j2 == j
                    if j2 == j:
                        der2 = pt * ((1 - pt)**gamma) * temp
                else:
                    temp1 = gamma * pt * math.log(pt) + pt - 1
                    temp2 = gamma * (1 - gamma) * pt * math.log(pt) + gamma * (1 - pt) * math.log(pt) + 2 * gamma * (1 - pt)
                    if j2 == target:
                    # j != target, j2 == target
                        der2 = -pj * pt * ((1 - pt)**(gamma -1)) * temp2 + pj**2 * pt * ((1 - pt)**(gamma -1)) * temp1
                    elif j2 != target and j2 != j:
                    # j != target, j2 != target, j2 != j
                        der2 = (pj * pj2 * ((1 - pt)**(gamma - 1))) * temp1 + (pt * pj * pj2 * ((1 - pt)**(gamma - 2))) * temp2
                    elif j2 == j:
                    # j != target, j2 == j
                        der2 = (pj * (pj - 1) * ((1 - pt)**(gamma - 1))) * temp1 + (pt * (pj**2) * ((1 - pt)**(gamma - 2))) * temp2
                hess_row.append(-der2 * class_w[j2]) # 乘上类别权重,catboost中默认损失函数前面没有负号，推导中多加了负号
            grad.append(-der1 * class_w[j])
            hess.append(hess_row)
        return (grad, hess)

if __name__ == "__main__":

    iris = load_iris()
    X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target, test_size=0.3)
    train_data = Pool(data=X_train , label=y_train)
    # If loss function is a user defined object, then the eval metric must be specified.
    params = {
        'eval_metric': 'MultiClass'
                }
    # Initialize CatBoostClassifier with custom `loss_function`
    model = CatBoostClassifier(**params, loss_function=MultiClassObjective(gamma=2))
    # Fit model
    model.fit(train_data)
    # Predict
    y_preds = model.predict(X_test,
                                prediction_type='Class')
    print(accuracy_score(y_test, y_preds))