# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:36:50 2024
@author: Sara
"""

import sklearn 
import pandas as pd
import os
from glob import glob
from sklearn.model_selection import GroupKFold, GridSearchCV, LeaveOneGroupOut,LeavePGroupsOut
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import Lasso , LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,f_classif,SelectFromModel,RFE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from xgboost import XGBClassifier
from feature_engine.selection import SelectBySingleFeaturePerformance,SmartCorrelatedSelection
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler,OneHotEncoder
import re 
# import tensorflow as tf
# from tensorflow.keras import layers, models
import seaborn as sns
import numpy as np
import shap
from scipy.stats import pearsonr,spearmanr
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

outputResultsDir = r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data\dataForML\features_modello_risultati"
datasetPath = r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/dataForML/allfeaturesForStatisticalAnalysis.xlsx"
CLASSES = ["control", "microcirculation"]
#CLASSES = ["control", "nonseptic"]
#CLASSES = ["sepsis", "nonseptic"]
K = 5
N_FEATURES = 12

def create_train_val_splits(X, y, groups):
  gkf = GroupKFold(n_splits=K)
  # split is made according subject_ids (group list)
  gkf.get_n_splits(X, y, groups)
  X_train_splits = []
  y_train_splits = []
  X_val_splits = []
  y_val_splits = []
  for train_index, val_index in gkf.split(X, y, groups):
      X_train, y_train = X.iloc[train_index], y.iloc[train_index]
      X_val, y_val = X.iloc[val_index], y.iloc[val_index]
      subjects_train, subjects_test = groups.iloc[train_index], groups.iloc[val_index]
      X_train_splits.append(X_train)
      y_train_splits.append(y_train)
      X_val_splits.append(X_val)
      y_val_splits.append(y_val)     
  return X_train_splits, y_train_splits, X_val_splits, y_val_splits


###### NORMALIZATION FUNCTIONS ######
def standard_scaling(train_set, test_set):
    scaler = StandardScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)
    return train_set, test_set

def robust_scaling(train_set, test_set):
    scaler = RobustScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)
    return train_set, test_set

###### FEATURE SELECTION FUNCTIONS ######

def anovaFeatureSelection(X_train, y_train, n_feat):
    # Rank and select features
    scaler = RobustScaler()
    X_train_std = scaler.fit_transform(X_train)
    #sel = SelectKBest(score_func = f_classif, k=n_feat).fit(X_train, y_train)
    sel = SelectKBest(score_func = sklearn.feature_selection.mutual_info_classif, k=n_feat).fit(X_train_std, y_train)
    selected_features = X_train.columns[sel.get_support()] 
    return selected_features.tolist()

def lassoFeatureSelection(X_train, y_train):
    selector = SelectFromModel(
    LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=10))
    scaler = RobustScaler()
    scaler.fit(X_train)
    selector.fit(scaler.transform(X_train), y_train)
    removed_feats = X_train.columns[(selector.estimator_.coef_ == 0).ravel().tolist()]
    selected_feats = [i for i in X_train.columns if i not in removed_feats]
    return selected_feats

def recursiveFeatureElimination(X_train, y_train, n_feat):
    scaler = RobustScaler()
    X_train_std = scaler.fit_transform(X_train)
    clf = RandomForestClassifier(n_estimators=10, random_state=10)
    sel_ = RFE(
        clf,
        n_features_to_select=n_feat,
        step=2,
    )
    sel_.fit(X_train_std, y_train)
    selected_features = selected_features = X_train.columns[sel_.support_]
    return selected_features.tolist()

def correlationWithLabel(X_train, n_feat):
    X_train.drop(columns=["R2_of_fit_gauss","SubjectId"], inplace=True)
    X_train.replace({'M': 0, 'F': 1},inplace=True)
    corrmat = X_train.corr(method='spearman')
    cor_target = abs(corrmat["Sex"])
    cor_target = cor_target[cor_target>0.5]
    return cor_target

def univariateFeatureSelection(X_train, y_train, n_feat):
    scaler = RobustScaler()
    X_train_std = scaler.fit_transform(X_train)
    sel = SelectBySingleFeaturePerformance(
    estimator=RandomForestClassifier(random_state=10),
    scoring='roc_auc',
    cv=3,
    threshold=None)
    X_train_t = sel.fit(X_train_std, y_train)
    sorted_features = pd.Series(sel.feature_performance_).sort_values(
    ascending=False)
    sorted_features.plot.bar(figsize=(10,5))
    plt.title("Univariate Feature Selection")
    plt.show()
    plt.close()
    numeric_indices = [int(re.search(r'\d+', col).group()) for col in sorted_features.index.tolist()]
    selected_features = X_train.columns[numeric_indices[:n_feat]].tolist()
    return selected_features

def correlatedFeaturesSelection(X_train,y_train):
    # scaler = RobustScaler()
    # X_train_std = scaler.fit_transform(X_train)
    tr = SmartCorrelatedSelection(
    variables=None,
    method="pearson",
    threshold=0.8,
    missing_values ="raise",
    #selection_method="model_performance",
    selection_method ="variance",
    estimator = None)
    #estimator = sklearn.tree.DecisionTreeClassifier(random_state=1))
    Xt = tr.fit(X_train)
    #Xt = tr.fit_transform(X_train, y_train)
    removed_feat = tr.features_to_drop_
    # numeric_indices = [int(re.search(r'\d+', col).group()) for col in removed_feat_idx]
    # removed_feat = X_train.columns[numeric_indices].tolist()
    selected_features = [i for i in X_train.columns if i not in removed_feat]
    return selected_features, removed_feat

###### TRAINING FUNCTIONS ######

def train_xgboost(X_train, X_test, y_train, y_test):
    # Creazione del modello XGBoost
    xgboost_model = XGBClassifier(n_estimators=150,learning_rate=0.1,max_depth=2)
    # Addestramento del modello
    trained_model = xgboost_model.fit(X_train, y_train)
    trainResults = accuracy_score(y_train,trained_model.predict(X_train))
    print(f"Training XGB accuracy : {trainResults}")
    y_pred = trained_model.predict(X_test)
    return trained_model, y_pred


def train_random_forest(X_train, X_test, y_train, y_test):
    # Creazione del modello RandomForest con iperparametri regolati
    random_forest_model = RandomForestClassifier(
        n_estimators=100,          # Numero di alberi
        max_depth=3,             # Profondità massima degli alberi
        # min_samples_split=5,      # Numero minimo di campioni richiesti per suddividere un nodo
        # min_samples_leaf=2,       # Numero minimo di campioni richiesti per essere una foglia
        max_features='sqrt',      # Numero di caratteristiche da considerare per il miglior split
        random_state=42
    )
    
    # Addestramento del modello
    random_forest_model.fit(X_train, y_train)
    
    # Accuratezza sul set di addestramento
    trainResults = accuracy_score(y_train, random_forest_model.predict(X_train))
    print(f"Training RF accuracy : {trainResults}")
    
    # Predizioni sul set di test
    y_pred = random_forest_model.predict(X_test)
    return random_forest_model, y_pred

###### MODEL EVALUATION ######
def create_confusion_matrix(y,y_pred,classLabel):
  cm = confusion_matrix(y, y_pred)
  df_cm = pd.DataFrame(cm, classLabel, classLabel)
  plt.figure(figsize = (10,6))
  conf = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Reds',annot_kws={"color": "black"})
  conf.set_xlabel('Prediction')
  conf.set_ylabel('True')
  plt.show()
  plt.close()
  print(cm)
  TP = cm[0][0]
  FN = cm[0][1]
  FP = cm[1][0]
  TN = cm[1][1]

  # Sensitivity, hit rate, recall, or true positive rate
  TPR = TP/(TP+FN)
  # Specificity or true negative rate
  TNR = TN/(TN+FP) 
  # Precision or positive predictive value
  PPV = TP/(TP+FP)
  # Negative predictive value
  NPV = TN/(TN+FN)
  # Fall out or false positive rate
  FPR = FP/(FP+TN)
  # False negative rate
  FNR = FN/(TP+FN)
  # Overall accuracy
  ACC = (TP+TN)/(TP+FP+FN+TN)
  # F1 score
  F1 = f1_score(y, y_pred, average='macro')
  print(f'Accuracy is :{100*ACC:0.2f}%')
  print(f'Sensitivity is : {100*TPR:0.2f}%')
  print(f'Specificity is {100*TNR:0.2f}%')  
  print(f'Precision is {100*PPV:0.2f}%')
  print(f'F1 Score is {100*F1:0.2f}%')
  metrics = {
        'Accuracy': ACC,
        'Sensitivity': TPR,
        'Specificity': TNR,
        'Precision': PPV,
        'Negative Predictive Value': NPV,
        'F1 Score': F1
    }
  
  return cm, metrics 

def create_confusion_3matrix(y, y_pred, class_labels):
    cm = confusion_matrix(y, y_pred)
    df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    plt.figure(figsize=(10,7))
    conf = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Reds',annot_kws={"color": "black"})
    conf.set_xlabel('Predicted Label')
    conf.set_ylabel('True Label')
    plt.show()
    plt.close()
    
    print("Confusion matrix:")
    print(cm)
    
    # Calcolo delle metriche per ogni classe
    metrics = {}
    for i, label in enumerate(class_labels):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (FP + FN + TP)
        
        TPR = TP / (TP + FN)  # Sensitivity, recall, or true positive rate
        TNR = TN / (TN + FP)  # Specificity or true negative rate
        PPV = TP / (TP + FP)  # Precision or positive predictive value
        NPV = TN / (TN + FN)  # Negative predictive value
        FPR = FP / (FP + TN)  # False positive rate
        FNR = FN / (TP + FN)  # False negative rate
        ACC = (TP + TN) / (TP + FP + FN + TN)  # Overall accuracy
        
        metrics[label] = {
            'Accuracy': ACC,
            'Sensitivity': TPR,
            'Specificity': TNR,
            'Precision': PPV,
            'Negative Predictive Value': NPV,
            'False Positive Rate': FPR,
            'False Negative Rate': FNR
        }
        
        print(f"\nMetrics for class '{label}':")
        print(f"Accuracy: {ACC:.2%}")
        print(f"Sensitivity (Recall): {TPR:.2%}")
        print(f"Specificity: {TNR:.2%}")
        print(f"Precision: {PPV:.2%}")
        print(f"Negative Predictive Value: {NPV:.2%}")
        print(f"False Positive Rate: {FPR:.2%}")
        print(f"False Negative Rate: {FNR:.2%}")
    
    return cm, metrics


def calculate_SHAP(fitted_model, X_train, X_test):
    #Fit the explainer
    explainer = shap.Explainer(fitted_model)
    
    #Evaluate SHAP values: calcola i valori SHAP per ogni fold su dati di TRAIN
    shap_values_train = explainer.shap_values(X_train)

    #Evaluate SHAP values: calcola i valori SHAP per ogni fold su dati di TEST
    shap_values_test = explainer.shap_values(X_test)
    #plotta i valori SHAP per visualizzare quali features impattano di più sul modello
    #shap.summary_plot(shap_values_test, X_test, feature_names= X.columns.values.tolist(),plot_type='bar')
    
    shap.summary_plot(shap_values_train, X_train, feature_names= X.columns.values.tolist(),plot_type='bar')
    
    return np.abs(shap_values_train).mean(axis=0), np.abs(shap_values_test).mean(axis=0)
##############################################

dataset = pd.read_excel(datasetPath)
X = dataset.drop(columns=["subject_id", "class"])

#FIND OPTIMAL NUMBER OF COMPONENTS FOR PCA

# Normalizza i dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # Applica PCA
# pca = PCA()
# pca.fit(X_scaled)

# # Calcola la varianza spiegata cumulativa
# explained_variance = np.cumsum(pca.explained_variance_ratio_)

# # Traccia la varianza spiegata cumulativa
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Explained Variance vs. Number of Components')
# plt.grid()
# plt.show()

y = dataset["class"]
subjects = dataset['subject_id']

features_importance = []

# set kfold crossvalidation
gkf = GroupKFold(n_splits=K)
# split is made according subject_ids
gkf.get_n_splits(X, y, subjects)
used_features = []
folds_result = {}

shap_train_values_all_folds = []
shap_test_values_all_folds = []

for i, (train_index, val_index) in enumerate(gkf.split(X, y, subjects)):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = X.iloc[val_index], y.iloc[val_index]
    subjects_train, subjects_test = subjects.iloc[train_index], subjects.iloc[val_index]
    print(f"Fold {i}")
    # clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    # models,predictions = clf.fit(X_train, X_val, y_train, y_val)
    # print(models)
    # # Select features
    features_corr, removed_feat_corr = correlatedFeaturesSelection(X_train, y_train)
    #features_anova = anovaFeatureSelection(X_train[features_corr],y_train,N_FEATURES)
    # features_anova = anovaFeatureSelection(X_train,y_train,N_FEATURES)
    used_features.extend(features_corr)
    X_train_s, X_val_s = X_train[features_corr], X_val[features_corr]
    #X_train_s, X_val_s = X_train, X_val
    # normalize data based on training set distribution
    X_train_std, X_val_std = robust_scaling(X_train_s, X_val_s)
    #train xgboost model
    print(f"XGBOOST \n")
    xgb_model, y_pred_xgb = train_xgboost(X_train_std, X_val_std, y_train, y_val)
    cm_xgb, metrics_xgb = create_confusion_matrix(y_val, y_pred_xgb, CLASSES)
    folds_result[f'Fold{i}'] = metrics_xgb
    shap_values_train_xgb, shap_values_test_xgb = calculate_SHAP(xgb_model, X_train_std, X_val_std)
    

    #train rf model
    # print(f"RandomForest \n")
    # rf_model, y_pred_rf = train_random_forest(X_train_std, X_val_std, y_train, y_val)
    # cm_rf, metrics_rf = create_confusion_matrix(y_val, y_pred_rf, CLASSES)
    # Apply PCA
    # print(f"RF with PCA\n")
    # pca = PCA(n_components=7)
    # X_train_pca = pca.fit_transform(X_train_std)
    # X_val_pca = pca.transform(X_val_std)
    # xgb_model_pca, y_pred_xgb_pca = train_xgboost(X_train_pca, X_val_pca, y_train, y_val)
    #cm_xgb_pca = create_confusion_matrix(y_val, y_pred_xgb_pca, CLASSES)
    # rf_model_pca, y_pred_rf_pca = train_random_forest(X_train_pca, X_val_pca, y_train, y_val)
    # cm_rf_pca, metrics_rf_pca = create_confusion_matrix(y_val, y_pred_rf_pca, CLASSES)
    #shap_values_train_xgb_pca, shap_values_test_xgb_pca = calculate_SHAP(xgb_model, X_train_pca, X_val_pca)


    

model_result = pd.DataFrame(folds_result)
model_result['Mean'] = model_result.mean(axis=1)
print(model_result['Mean'])
model_result.to_excel(os.path.join(outputResultsDir,f"xgboost_n_est150,lr0.1,m_depth2_pearson07.xlsx"))

    
    
    
    
    