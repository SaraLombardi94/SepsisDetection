# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:13:08 2023
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
import tensorflow as tf
from tensorflow.keras import layers, models
import seaborn as sns
import numpy as np
import shap
from scipy.stats import pearsonr,spearmanr
from collections import Counter

modelData = pd.read_excel(r"D:/progettoSessoEtaPPG/fit2024/dataset/expFeatures.xlsx")
X = modelData.drop(columns=["R2_of_fit","SubjectId","Sex","dicnotch","Age"])
y = modelData["Sex"]
# obtain numerical values for labels 
y = y.replace({'M': 0, 'F': 1})

subjects = modelData["SubjectId"]
N_FEATURES = 12
CLASSES = ['Control','Sepsis']

## Function to normalize signal ##
def normalizeData(train_set, test_set):
    scaler = StandardScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)
    return train_set, test_set

def robust_scale_features(train_set, test_set):
    scaler = RobustScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)
    return train_set, test_set

###### FEATURE SELECTION FUNCTIONS  ######

def anovaFeatureSelection(X_train, y_train, n_feat):
    # Rank and select features
    scaler = RobustScaler()
    X_train_std = scaler.fit_transform(X_train)
    # sel = SelectKBest(score_func = f_classif, k=n_feat).fit(X_train, y_train)
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

def correlatedFeaturesSelection(X_train):
    # scaler = RobustScaler()
    # X_train_std = scaler.fit_transform(X_train)
    tr = SmartCorrelatedSelection(
    variables = None,
    method = "spearman",
    threshold = 0.8,
    missing_values ="raise",
    selection_method="variance",
    estimator = None)
    Xt = tr.fit(X_train)
    removed_feat = tr.features_to_drop_
    # numeric_indices = [int(re.search(r'\d+', col).group()) for col in removed_feat_idx]
    # removed_feat = X_train.columns[numeric_indices].tolist()
    selected_features = [i for i in X_train.columns if i not in removed_feat]
    return selected_features, removed_feat

def r_pvalues(df):
    cols = pd.DataFrame(columns=df.columns)
    p = cols.transpose().join(cols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            p[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    p.replace(0, np.nan, inplace=True)
    return p.astype(float)

###### MODELS TRAINING FUNCTIONS ######

def train_adaboost(X_train, X_test, y_train, y_test):
    # Creazione del modello AdaBoost
    # adaboost_model = AdaBoostClassifier(n_estimators=100,learning_rate=0.1,base_estimator=DecisionTreeClassifier(max_depth=1), algorithm='SAMME.R',random_state=42)
    # Addestramento del modello
    adaboost_model = AdaBoostClassifier(random_state=42)
    adaboost_model = adaboost_model.fit(X_train, y_train)
    trainResults = accuracy_score(y_train,adaboost_model.predict(X_train))
    print(f"Training AB accuracy : {trainResults}")
    y_pred = adaboost_model.predict(X_test)
    return adaboost_model, y_pred

def gridSearchAdaBoostKfold(X, y, subjects):
    # Inizializza il tuo scalatore
    scaler = RobustScaler()
    X_std = scaler.fit_transform(X)
    
    # Inizializza una lista per memorizzare i migliori parametri trovati in ciascuna iterazione di LOSO-CV
    migliori_parametri_lista = []

    # Inizializza KFold con raggruppamento per soggetto
    gkf = GroupKFold(n_splits=10)
    indices= []
    for train_index, test_index in gkf.split(X, y, groups=subjects):
        indices.append((train_index,test_index))

    # Definisci la griglia dei parametri da testare
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'base_estimator': [None, DecisionTreeClassifier(max_depth=1), RandomForestClassifier(n_estimators=50)],
        'algorithm': ['SAMME', 'SAMME.R']
        # Altri parametri di AdaBoost che desideri ottimizzare
    }

    # Definisci il classificatore AdaBoost
    adaboost = AdaBoostClassifier()

    # Crea l'oggetto GridSearchCV con la cross-validazione interna
    grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=indices, scoring='f1_micro')

    # Esegui la ricerca a griglia sul training set
    grid_search.fit(X_std, y)

    # Visualizza i parametri ottimizzati
    print("Parametri ottimizzati per la corrente iterazione di KFold:", grid_search.best_params_)

    # Aggiungi i migliori parametri alla lista
    migliori_parametri_lista.append(grid_search.best_params_)

    # Restituisci la lista di migliori parametri trovati in ogni iterazione di KFold
    return migliori_parametri_lista


def gridSearchXGBoost(X_train, y_train):
    scaler = RobustScaler()
    X_train_std = scaler.fit_transform(X_train)
    param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 4],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.8, 1.0],
    }
    # Definisci il classificatore XGBoost
    xgb_classifier = XGBClassifier()
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='f1_micro')
    # Esegui la ricerca a griglia con cross-validazione
    grid_search.fit(X_train_std, y_train)
    print("Parametri ottimizzati XGBoost:", grid_search.best_params_)
    return grid_search.best_params_

def gridSearchXGBoostSubjects(X, y, subjects):
    param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 4],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.8, 1.0]
    }

    # Definisci il classificatore XGBoost
    xgb_classifier = XGBClassifier()

    # Utilizza GroupKFold per la cross-validazione basata su soggetti
    group_kfold = GroupKFold(n_splits=3)

    grid_search = GridSearchCV(
        estimator=xgb_classifier,
        param_grid=param_grid,
        cv=group_kfold.split(X, y, groups=subjects),
        scoring='f1_micro'
    )

    # Esegui la ricerca a griglia con cross-validazione
    grid_search.fit(X, y)

    print("Parametri ottimizzati XGBoost:", grid_search.best_params_)
    return grid_search.best_params_


def train_xgboost(X_train, X_test, y_train, y_test):
    # Creazione del modello XGBoost
    #xgboost_model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=4,subsample=0.8,colsample_bytree=0.8, random_state=42)
    #xgboost_model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=4,subsample=0.8,colsample_bytree=0.8, random_state=42)
    #xgboost_model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,subsample=0.8, random_state=42)
    xgboost_model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
    # Addestramento del modello
    trained_model = xgboost_model.fit(X_train, y_train)
    trainResults = accuracy_score(y_train,trained_model.predict(X_train))
    print(f"Training XGB accuracy : {trainResults}")
    y_pred = trained_model.predict(X_test)
    return trained_model, y_pred

def train_random_forest(X_train, X_test, y_train, y_test):
    # Creazione del modello RandomForest
    random_forest_model = RandomForestClassifier(n_estimators=50, random_state=42)
    # Addestramento del modello
    random_forest_model.fit(X_train, y_train)
    trainResults = accuracy_score(y_train,random_forest_model.predict(X_train))
    print(f"Training RF accuracy : {trainResults}")
    y_pred = random_forest_model.predict(X_test)
    return random_forest_model, y_pred

##### MODEL EVALUATION ####

def create_confusion_matrix(y,y_pred, classes):
  cm = confusion_matrix(y, y_pred)
  df_cm = pd.DataFrame(cm, classes, classes)
  plt.figure(figsize = (10,6))
  conf = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
  conf.set_xlabel('Prediction')
  conf.set_ylabel('True')
  plt.show()
  plt.close()

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
  print(f'Accuracy is :{100*ACC:0.2f}%')
  print(f'Sensitivity is : {100*TPR:0.2f}%')
  print(f'Specificity is {100*TNR:0.2f}%')  
  print(f'Precision is {100*PPV:0.2f}%')
  return cm

###### Feature importance statistics ######

def calculate_SHAP(fitted_model, X_train, X_test):
    #Fit the explainer
    explainer = shap.Explainer(fitted_model)
    
    #Evaluate SHAP values: calcola i valori SHAP per ogni fold su dati di TRAIN
    shap_values_train = explainer.shap_values(X_train)

    #Evaluate SHAP values: calcola i valori SHAP per ogni fold su dati di TEST
    shap_values_test = explainer.shap_values(X_test)
    #plotta i valori SHAP per visualizzare quali features impattano di più sul modello
    # prende del tempo
    #shap.summary_plot(shap_values_test, X_test, feature_names= X.columns.values.tolist(),plot_type='bar')
    
    #shap.summary_plot(shap_values_train, X_train, feature_names= X.columns.values.tolist(),plot_type='bar')
    
    return np.abs(shap_values_train).mean(axis=0), np.abs(shap_values_test).mean(axis=0)    

loso_ada = []
loso_xgb = []
tested_sub = []
tested_sub_label = []
features_importance = []
shap_train= {}
shap_test = []
used_features = []
parameters_grid = []
# create train/test splits for crossvalidation
gkf = GroupKFold(n_splits=len(np.unique(np.array(subjects))))
for i, (train_index, test_index) in enumerate(gkf.split(X, y, subjects)):
    
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    subjects_train, subjects_test = subjects.iloc[train_index], subjects.iloc[test_index]
    # rimuove le features altamente correlate tra loro 
    features_corr, removed_feat_corr = correlatedFeaturesSelection(X_train)
    # seleziona k-best features 
    features_anova = anovaFeatureSelection(X_train[features_corr],y_train,N_FEATURES)
    # salvo le features usate in questa iterazione
    used_features.extend(features_anova)
    X_train_s, X_test_s = X_train[features_anova], X_test[features_anova]
    
    # normalize data based on training set distribution
    X_train_std, X_test_std = robust_scale_features(X_train_s, X_test_s)
  
    xgb_model, y_pred_xgb = train_xgboost(X_train_std, X_test_std, y_train, y_test)
    # per xgboost se vuoi puoi salvare le features più importanti 
    features_importance.append(xgb_model.feature_importances_)
    # plot
    # plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
    # plt.show()
    print(f"Classification Report XGBoost: \n", accuracy_score(y_test, y_pred_xgb))
    # print(f"Classification Report XGBoost:\n method {m} \n", classification_report(y_test, y_pred_xgb))
    loso_xgb.append(accuracy_score(y_test, y_pred_xgb))
    # TODO: salva i risultati per ogni fold 

usedFeaturesStats = Counter(used_features)
