import numpy as np
import torch
from logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(embeds, idx_train, idx_test, label, nb_classes, device, lr, wd):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    test_embs = embeds[idx_test]

    train_lbls = label[idx_train]
    test_lbls = label[idx_test]


    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
    log.to(device)

    test_acc_list = []

    for iter_ in range(200):
        # train
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        preds = torch.argmax(logits, dim=1)
        train_acc = torch.sum(preds == train_lbls).float() / train_lbls.shape[0]

        loss.backward()
        opt.step()

        # -1
        log.eval()
        logits = log(test_embs)

        preds = torch.argmax(logits, dim=1)
        test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]

        test_acc_list.append(test_acc.cpu().numpy())

        print('epoch: ', iter_, ' train_acc: ', train_acc, ' test_acc: ', test_acc)

    return test_acc_list


