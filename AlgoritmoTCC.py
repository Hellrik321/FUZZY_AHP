# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:13:13 2020

@author: wendel
"""

# Declaração de Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from func import salvar_grafico, gerar_csv
from math import sqrt
from statistics import mean, stdev
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier

# Importação do banco de dados
dados = pd.read_csv(r"D:/Users/phlo3/Documents/Unifesspa/Projeto de Pesquisa/Algoritmo_TCC-master/Algoritmo_TCC-master/Banco1.csv")

# Separação de atributos e labels
previsores = dados.iloc[:, 0:8].values
classes = dados.iloc[:, 8].values

# Padronização
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Clusterização
cluster = KMeans(n_clusters=3, random_state=0)
model = cluster.fit(previsores)
classes = model.labels_

# Criação de variáveis para gerar arquivo csv para exportação de dados
Acu_csv = pd.DataFrame()
F1_csv = pd.DataFrame()
Prec_csv = pd.DataFrame()
FPR_csv = pd.DataFrame()
FNR_csv = pd.DataFrame()
Sens_csv = pd.DataFrame()
Espec_csv = pd.DataFrame()

#Instanciamento dos classificadores
Dummy = DummyClassifier()
NaiveBayes = GaussianNB()
RegressaoLog = LogisticRegression()
SVM = SVC(kernel = 'rbf', random_state = 1, C = 2.0, probability=True)
MLP = MLPClassifier(max_iter = 500, solver='lbfgs', hidden_layer_sizes=(100), 
                     activation = 'identity', batch_size=100, learning_rate_init=0.1)
RandomForest = RandomForestClassifier(n_estimators=100, criterion='gini')

#Variáveis auxiliares
metodos = (RegressaoLog, RandomForest, MLP, SVM, NaiveBayes, Dummy)
metodos2 = ['L.Regression', 'R.Forest', 'MLP', 'SVM', 'NaiveBayes', 'Dummy']
cont = 0
int_conf = pd.DataFrame()
metricas = pd.DataFrame()
count = 0

for m in metodos:
    
    # Variáveis auxiliares
    acuracia_total = []
    f1_total = []
    precisao_total = []
    fpr_total = []
    fnr_total = []
    sens_total = []
    espec_total = []
    gmean_total = []
    matr1 = []
    matr2 = []
    matr3 = []
    matr4 = []
    contador = 0
    
    for c in range(30):
        # Divisão em treinamento e teste
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=c)
        
        # Variáveis auxiliares
        iteracao2 = []
        f1 = []
        precisao = []
        fnr_parc = []
        fpr_parc = []
        sens_parc = []
        espec_parc = []
        gmean_parc = []

        # Instanciamento dos classificadores
        for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(classes.shape[0], 1))):
            
            # Treinamento dos classificadores
            m.fit(previsores[indice_treinamento], classes[indice_treinamento])
            xyz = classes[indice_teste]
            previsoes = m.predict(previsores[indice_teste])

            # Análise dos resultados
            acuracia = accuracy_score(classes[indice_teste], previsoes)*100
            
            f1_parcial = f1_score(classes[indice_teste], previsoes, average='micro')*100
            precisao_parcial = precision_score(classes[indice_teste], previsoes, average='micro')*100
            
            
            
            
            
            # Preenchimento dos vetores de dados
            iteracao2.append(acuracia)
            f1.append(f1_parcial)
            precisao.append(precisao_parcial)
            
            
            

        iteracao2 = np.asarray(iteracao2)
        media = iteracao2.mean()
        acuracia_total.append(media)
        
        f1 = np.asarray(f1)
        mediaf1 = f1.mean()
        f1_total.append(mediaf1)
        
        precisao = np.asarray(precisao)
        mediaprec = precisao.mean()
        precisao_total.append(mediaprec)
        
        
        
        
        
        
        
        
        
        
        
    Acu_csv[m] = acuracia_total
    F1_csv[m] = f1_total
    Prec_csv[m] = precisao_total
    
    Sens_csv[m] = sens_total
    Espec_csv[m] = espec_total
    
    
    # Plotagem da matriz de confusão
    
    
    cont+=1

    # Valores das métricas com intervalo de confiança
    teste = (acuracia_total, f1_total, precisao_total)
    for t in teste:
        int_conf.loc[metodos2[count], contador] = stdev(t)
        metricas.loc[metodos2[count], contador] = round(mean(t), 2)
        contador+=1
    
    count+=1

# Renomeando as colunas das matrizes das métricas
int_conf.columns = ['Acurácia', 'F1', 'Precisão']
metricas.columns = ['Acurácia', 'F1', 'Precisão']

# Chamada das funções de plotagem dos gráficos das métricas e de geração dos arquivos CSV com os dados 
salvar_grafico(metricas, int_conf)
csvs = (Acu_csv, F1_csv, Prec_csv)
for doc in csvs:
    gerar_csv(doc)


