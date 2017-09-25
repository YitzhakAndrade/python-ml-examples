import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn import tree

#Vamos usar with pois é legal fechar o documento depois da operaçao
with open("titanic3.csv",'r') as csvfile:
    titanic_reader = csv.reader(csvfile,delimiter=',',quotechar='"')
    
    #row agora contem os nomes dos atributos
    row = titanic_reader.__next__()
    feature_names = np.array(row)
    
    #Carrega o conjunto de treino e as classes
    titanic_X,titanic_Y = [],[]
    for row in titanic_reader:
        titanic_X.append(row)
        titanic_Y.append(row[1])#A classe é "survived"
        
    titanic_X = np.array(titanic_X)
    titanic_Y = np.array(titanic_Y)
    
    #Vamos selecionar os atributos para treino agora.
    titanic_X = titanic_X[:,[0,3,4]]
    feature_names = feature_names[[0,3,4]]

    ages = titanic_X[:,2]
    
    mean = np.mean(np.array([x for x in ages if x != '']).astype(float))
    titanic_X[titanic_X[:,2] == '',2] = mean
    
    #Carregamos a classe LabelEncoder
    enc = LabelEncoder()
    
    #Agora associamos isso a coluna correspondente
    titanic_X[:,1] = enc.fit_transform(titanic_X[:,1])
    
    #Vamos preparar os conjuntos de treino e de teste
    x_train,x_test,y_train,y_test = train_test_split(
            titanic_X,titanic_Y,
            test_size=0.25,
            random_state=33)
    
    clf = tree.DecisionTreeClassifier(
            criterion='entropy',
            max_depth=3,
            min_samples_leaf=5)
    
    clf = clf.fit(x_train,y_train)
    
    
    def makeAcuracy(tree,x_test,y_test):
        predictions = clf.predict(x_test)
        erro = 0.0
        for x in range(len(predictions)):
            if predictions[x] != y_test[x]:
                erro += 1.
        acuracy = (1-(erro/len(predictions)))
        return acuracy,predictions
    
    acur, pred = makeAcuracy(clf,x_test,y_test)
    print(acur)
    
    
    #print(  clf.predict([1,0,29])   )
    #print(  clf.predict([1,1,99])   )