# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:51:35 2024

@author: Utente
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter
from feature_engine.selection import SmartCorrelatedSelection
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns

def load_dataset(file_path):
    """Carica il dataset da un file Excel."""
    return pd.read_excel(file_path)

def fill_missing_values(data):
    """Riempi i valori mancanti con la media delle colonne numeriche."""
    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    return data


####### trova la correlazione tra le features #######

def correlatedFeaturesSelection(X_train):
    # Inizializza l'oggetto SmartCorrelatedSelection
    tr = SmartCorrelatedSelection(
        variables=None,
        method="spearman",
        threshold=0.8,
        missing_values="raise",
        selection_method="variance",
        estimator=None
    )
    
    # Adatta il selettore ai dati di addestramento
    Xt = tr.fit(X_train)
    
    # Ottieni le features rimosse
    removed_feat = tr.features_to_drop_
    
    # Ottieni le features selezionate
    selected_features = [i for i in X_train.columns if i not in removed_feat]
    
    return selected_features, removed_feat


###### test statistico per selezionare le k best features ######

def perform_anova_test(x_train,y_train, k_features):
    """Esegui il test ANOVA e seleziona le k migliori features."""
    # Seleziona le k migliori features utilizzando il test ANOVA
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Ottieni gli indici delle features selezionate
    selected_indices = selector.get_support(indices=True)

    # Ottieni i nomi delle features selezionate
    selected_features = X_train.columns[selected_indices].tolist()

    return selected_features

def calculate_feature_percentage(significant_features_per_fold, total_folds):
    """Calcola la percentuale di volte in cui ciascuna feature è stata considerata significativa."""
    feature_counts = Counter(feature for fold_features in significant_features_per_fold for feature in fold_features)
    feature_percentages = {feature: (count / total_folds) * 100 for feature, count in feature_counts.items()}
    return feature_percentages

def plot_feature_significance(feature_percentage_dict):
    """Rappresenta graficamente l'utilizzo delle features."""
    features = list(feature_percentage_dict.keys())
    percentages = list(feature_percentage_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(features, percentages, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Percentage of Significance')
    plt.title('Percentage of Significance for Each Feature')
    plt.xticks(rotation=90)  # Ruota le etichette sull'asse x per una migliore leggibilità
    plt.tight_layout()
    plt.show()


#normalizzazione delle features

def robust_scale_features(train_set, test_set):
    scaler = RobustScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)
    return train_set, test_set

#modello di ML
def train_xgboost(X_train, X_test, y_train, y_test):
    # Creazione del modello XGBoost
    #xgboost_model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=4,subsample=0.8,colsample_bytree=0.8, random_state=42)
    #xgboost_model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=4,subsample=0.8,colsample_bytree=0.8, random_state=42)
    #xgboost_model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,subsample=0.8, random_state=42)
    xgboost_model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,subsample=0.8, random_state=42)
    # Addestramento del modello
    trained_model = xgboost_model.fit(X_train, y_train)
    trainResults = accuracy_score(y_train,trained_model.predict(X_train))
    #print(f"Training XGB accuracy : {trainResults}")
    y_pred = trained_model.predict(X_test)
    # Calcola l'accuratezza delle previsioni
    test_accuracy = accuracy_score(y_test, y_pred)
    #print(f"Test accuracy: {test_accuracy}")
    return trained_model, y_pred



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
  #print(f'Accuracy is :{100*ACC:0.2f}%')
  #print(f'Sensitivity is : {100*TPR:0.2f}%')
  #print(f'Specificity is {100*TNR:0.2f}%')  
  #print(f'Precision is {100*PPV:0.2f}%')
  return cm



# Carica il dataset
df = load_dataset(r'C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/all_features_with_subid.xlsx')
# Riempi i valori mancanti con la media della colonna
fill_missing_values(df)
X = df.drop(columns=['label', 'nome_segnale', 'SubId','R2_of_fit'])
y = df['label']
groups = df['SubId']  # Colonna SubId per i gruppi


# Definisci il numero di fold
k = 25
# Definisci il numero di features che voglio utilizzare
k_features = 15

# Inizializza GroupKFold
group_kfold = GroupKFold(n_splits=k)

selected_features_per_fold = []  # Lista vuota per memorizzare le feature selezionate con correlatedfeaturesselection per ogni fold
removed_features_per_fold = []  # Lista vuota per memorizzare le feature rimosse con correlatedfeaturesselection per ogni fold
significant_features_per_fold = []  # Lista vuota per memorizzare le feature significative per ogni fold
features_importance = []

# Liste vuote per memorizzare le metriche di valutazione di ogni fold
accuracy_scores = []
sensitivity_scores = []
specificity_scores = []
precision_scores = []


# Ciclo su ogni fold per eseguire la cross-validation con tenuta
for fold, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

##### RIMOZIONE DELLE FEATURES ALTAMENTE CORRELATE TRA LORO ######

    # Rimuovi le features altamente correlate
    selected_features, removed_features = correlatedFeaturesSelection(X_train)

##### SELEZIONO LE FEATURES PIù IMPORTANTI #####

    # Utilizza le features selezionate per il test ANOVA
    X_train_selected = X_train[selected_features]

    # Esegui il test ANOVA sui dati di training senza le features altamente correlate
    significant_features = perform_anova_test(X_train_selected,y_train, k_features)

    # Riscrivo i dati di training e test con i risultati dell'ANOVA
    X_train_s, X_test_s = X_train[significant_features], X_test[significant_features]

#### NORMALIZZO LE FEATURES ####

    # normalize data based on training set distribution
    X_train_std, X_test_std = robust_scale_features(X_train_s, X_test_s)

#### SCELTA DEGLI IPERPARAMETRI PER OGNI FOLD ####

    #trovo gli iperparametri migliori per ogni fold
    #best_params = gridSearchXGBoostSubjects(X_train_std, y_train, groups[train_index])
    #print("Best parameters for fold:", best_params)
    
#### ADDESTRAMENTO DEL MODELLO ####
    
    # Addestro e testo il modello per i dati di test
    xgb_model, y_pred_xgb = train_xgboost(X_train_std, X_test_std, y_train, y_test)
    
#### VALUTAZIONE DEL MODELLO #####    
    
    # Calcolo della matrice di confusione e valutazione delle prestazioni
    cm = create_confusion_matrix(y_test, y_pred_xgb, classes=['Class 0', 'Class 1'])
    TP, FN, FP, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    ACC = (TP + TN) / (TP + FP + FN + TN)


#### AGGIUNTA DELLE VARIABILI DA REGISTRARE, OGNUNA NELLA PROPRIA LISTA ######

    # Aggiungi le metriche di valutazione alle rispettive liste
    accuracy_scores.append(ACC)
    sensitivity_scores.append(TPR)
    specificity_scores.append(TNR)
    precision_scores.append(PPV)
    # Aggiungi le feature selezionate per questo fold alla lista
    selected_features_per_fold.append(selected_features)
    # Aggiungi le feature eliminate per questo fold alla lista
    removed_features_per_fold.append(removed_features)
    # Aggiungi le feature significative per questo fold alla lista
    significant_features_per_fold.append(significant_features)
    
#FINE DEL CICLO FOR CHE ITERA SU OGNI FOLD



###### RAPPRESENTAZIONE GRAFICA DELLE FEATURES PIU IMPORTANTI #####
# Calcola la frequenza di ciascuna feature in tutte le fold
total_folds = len(significant_features_per_fold)
feature_percentage_dict = calculate_feature_percentage(significant_features_per_fold, total_folds)

# Estrai il nome di tutte le features dal DataFrame df
all_features = df.drop(columns=['label', 'SubId', 'nome_segnale']).columns.tolist()

# Aggiungi le features non considerate significative con valore 0% al dizionario
for feature in all_features:
    if feature not in feature_percentage_dict:
        feature_percentage_dict[feature] = 0.0

# Rappresenta graficamente l'utilizzo delle features
plot_feature_significance(feature_percentage_dict)




#### VALUTAZIONE DELLE PRESTAZIONI MEDIE DEL MODELLO ####

# Calcola la media delle metriche di valutazione
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
average_sensitivity = sum(sensitivity_scores) / len(sensitivity_scores)
average_specificity = sum(specificity_scores) / len(specificity_scores)
average_precision = sum(precision_scores) / len(precision_scores)

# Stampa le metriche di valutazione complessive
print(f"Average Accuracy: {average_accuracy}")
print(f"Average Sensitivity: {average_sensitivity}")
print(f"Average Specificity: {average_specificity}")
print(f"Average Precision: {average_precision}")