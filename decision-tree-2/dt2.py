# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:44:11 2017
 
@author: julio
"""
 
import pandas as pd #Analise e processamento de dados
from sklearn.preprocessing import LabelEncoder #Para transformar tipos de dados
from sklearn.tree import DecisionTreeRegressor #O algoritmo DT para regressão
from sklearn.cross_validation import train_test_split #Preparar o dataset para treino e teste
import matplotlib.pyplot as plt #Fazer alguns graficos
from sklearn.metrics import mean_absolute_error
 
#Vamos começar abrindo o documento
#Usarei pandas uma lib magnifica para analise e
#preparação de dados.
#Sempre olhar o tipo de separador na hora de abrir o documento.
data = pd.read_csv("dataset_Facebook.csv",sep=";")
 
#vamos colocar os nomes dos atributos em uma coluna separada
#isso é muito util para relaizarmos alguns procedimentos.
#O pandas tem uma função chamada hasnans()
#É util para procurar valores faltando
ft = data.columns
 
#Antes de usar o conjunto de dados irei eliminar alguns
#Atributos que não irei usar para treino.
data = data.drop(ft[7:15],axis=1)
 
#Agora irei redefinir a variavel ft para incluir apenas
#os atributos que iremos usar
ft = data.columns
 
#Irei normalizar os dados da coluna page total likes.
#Apesar de não ser necessário para Dt isso nos ajuda diminuir
#o peso desse atributo e lidar com valores entre 0 e 1 é bem melhor para plotar
#Fazer um demonstração da função .apply
#Mas antes vamos definir algumas variaveis para ficar mais didaticos
maxi = data[ft[0]].max()
mini = data[ft[0]].min()
 
data[ft[0]] = data[ft[0]].apply(lambda x:(float(x) - mini)/(maxi-mini))
 
#Agora vamos cuidar dos valores faltando na coluna Paid
data[ft[6]] = data[ft[6]].fillna(data[ft[6]].median())
 
#Irei substituir os valores vazios na coluna like pela media
#Existem métodos mais apropriados
data[ft[8]] = data[ft[8]].fillna(data[ft[8]].mean())
 
#Irei substituir os valores da coluna Share pela média
data[ft[9]] = data[ft[9]].fillna(data[ft[9]].median())
 
#Agora iremos passar a coluna Type para valores discretos.
#Usaremos a classe LabelEncoder()
le = LabelEncoder()
data[ft[1]] = le.fit_transform(data[ft[1]])
 
#Agora iremor separarar conjunto de treino e teste
x_train,x_test,y_train,y_test = train_test_split(data[ft[:-1]].values,
data[ft[-1]].values,
train_size=0.85)
 
regr = DecisionTreeRegressor(criterion='mae')
regr.fit(x_train,y_train)
 
y = regr.predict(x_test)
 
plt.scatter(x_test[:,0],y,color="red")
plt.scatter(x_test[:,0],y_test)
plt.legend(['Predicoes','valores reais'])
 
plt.plot(range(len(y)),y,color="red")
plt.scatter(range(len(y)),y_test)
plt.legend(["predicoes","Valor real"])
 
loss = mean_absolute_error
 
loss(y_test,y)