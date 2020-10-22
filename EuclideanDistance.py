import joblib as jb
import numpy as np
from itertools import combinations
from scipy import spatial 
from tqdm import tqdm
from statistics import mean
from sklearn import metrics
import matplotlib.pyplot as plt
import random
random.seed(6)

file = jb.load('/Users/evanli/Downloads/SRP/VSCODE/all_features.joblib')
y_s = []
scores = []
genuine = []
imposter = []

run_limit = 50000

def score():
    paired_keys = list(combinations(list(file.keys()), 2))
    random.shuffle(paired_keys)
    run = 0
    for j in tqdm(paired_keys):
        if run == run_limit:
            break
        eye1 = file[j[0]]
        eye2 = file[j[1]]
        if len(eye1) == 0 or len(eye2) == 0:
            continue 
        dist = []
        for x in range(len(eye1)):
            for y in range(len(eye2)):
                dist.append(spatial.distance.euclidean(eye1[x][0], eye2[y][0]))
        dist = list(filter(lambda x: x<=12, dist))
        score = 0
        for distance in dist:
            if distance < 9:
                score+=50.0
            if distance < 10:
                score+=5.0
            if distance < 11:
                score+=1.0
            if distance < 12:
                score+=0.10
        if j[0][:6] == j[1][:6]:
            y_s.append(0)
            scores.append(score)
            genuine.append(score)
        else:
            y_s.append(1)
            scores.append(score)
            imposter.append(score)
        run+=1

def ROC(): 
    fpr, tpr, thresholds = metrics.roc_curve(y_s, scores, pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)
    print(fpr)
    print(tpr)
    print(thresholds)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

score()
print()
print(len(genuine), mean(genuine))
print(len(imposter), mean(imposter))

ROC()