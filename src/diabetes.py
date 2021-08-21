import os
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

plt.style.use('ggplot') #estilo de visualizacao

#LEITURA DOS DIRETORIOS

#retorna o local absoluto do diretório corrente da aplicação
#os.path.abspath('.')
#os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#join para juntar o BASE_DIR com seu diretorio filho
DATA_DIR = os.path.join(BASE_DIR, 'data')
#print(DATA_DIR)

#pegar o arquivo .csv
#file_name = os.listdir(DATA_DIR) # lista o conteudo do diretório
#my_files = []
#for i in file_name:
#    if i.endswith('.csv'):
#        my_files.append(i)

# LIST COMPREENSIONS DO DATASET
file_names = [i for i in os.listdir(DATA_DIR) if i.endswith('.csv')]

# df = dataframe - listando
for i in file_names:
    df = pd.read_csv(os.path.join(DATA_DIR, i))

# apresentado as informações do Dataset
# print('\n **************** Informações sobre o DataSet *************** \n')
# print('Diretorios: \n')
# print('Meu diretorio do projeto: \n', BASE_DIR)
# print('Meu diretorio de dados: \n', DATA_DIR)
# print('Este é meu dataset: \n', df.head(5))

#iniciando o tratamento dos dados... True = 1 e False = 0
map_data = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(map_data) #mapeando
#print('\n Alteração de valores categóricos: \n', df.head(5))

#numpy e pandas - 
sample0 = np.where(df.loc[df['diabetes'] == 0])
sample1 = np.where(df.loc[df['diabetes'] == 1])
#print('\nAmostra da classe 0 - Controle: ', sample0)
#print('\nAmostra da classe 1 - Paciente: ', sample1)

#quantidade de amostras por classe
vl_paciente = len(df.loc[df['diabetes'] == 1])
vl_controle = len(df.loc[df['diabetes'] == 0])
#print('\nAmostra da classe 0 - Controle: ', vl_controle)
#print('\nAmostra da classe 1 - Paciente: ', vl_paciente)

# verificar se existe dados faltantes no conjunto de dados
dt_feature = df.iloc[:, : -1]
dt_target = df.iloc[:, -1]
dt_feature = dt_feature.mask(dt_feature == 0).fillna(dt_feature.mean)
#print(dt_feature)
#print(dt_target)
#print('\n ************ Tratando dados iuais a 0 *************')
#print('colunas de valores igual a 0:\n', (df==0).sum())



#plotando dados 
#plt.style.use('ggplot') #estilo de visualizacao
def plot_hist():
    #histograma de classes
    plt.hist(df.iloc[:,-1], color='b', width=.1)
    plt.xlabel('Qtd Amostra')
    plt.ylabel('Hist da Classe')
    plt.show()

#hitograma web offline [verificar erro na execucao]
def target_count():
    trace = go.Bar(x = df['diabetes'].value_counts().values.tolist(),
                y = ['saudaveis', 'diabeticos'],
                orientation = 'v',
                text = df['diabetes'].value_counts().values.tolist(),
                textfont = dict(size=15),
                textposition = 'auto',
                opacity = 0.8, marker=dict(color=['lightskyblue', 'gold'],
                line=dict(color='#000000', width=1.5)))
    layout = dict(title='resultado')
    fig = dict(data=[trace], layout = layout)
    py.iplot(fig)


#analise de correlacao
def correlation(size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


#funcao bloxspot
def bloxplot():
    f, ax = plt.subplots(figsize=(11, 15))
    ax.set_facecolor('#fafafa')
    ax.set(xlim=(-0.5, 200))
    plt.ylabel('variaveis')
    plt.title('Overview Dataset')
    ax = sns.boxplot(data=df, orient='v', palette='Set2')
    plt.show()


#criando a funcao pra trazer todas as infromacaoes que esta sendo imprimida no console
def information():
    # apresentado as informações do Dataset
    print('\n **************** Informações sobre o DataSet *************** \n')
    print('Diretorios: \n')
    print('Meu diretorio do projeto: \n', BASE_DIR)
    print('Meu diretorio de dados: \n', DATA_DIR)
    print('Este é meu dataset: \n', df.head(5))

    print('\nAmostra da classe 0 - Controle: ', sample0)
    print('\nAmostra da classe 1 - Paciente: ', sample1)

    print('\nAmostra da classe 0 - Controle: ', vl_controle)
    print('\nAmostra da classe 1 - Paciente: ', vl_paciente)

    #verificar se existe dados faltantes no conjunto de dados
    print('\n ************ Tratando dados iuais a 0 *************')
    print('colunas de valores igual a 0:\n', (df==0).sum())
    print('o conjunto de dados possui: %d linhas e %d colunas para : '%(len(df[:]), len(df.columns)))
    print('  %d paciente, que corresponde a %.2f%% do conjunto de dados' %(vl_paciente, vl_paciente /(vl_paciente + vl_controle)*100))
    print('  %d controle, que corresponde a %.2f%% do conjunto de dados' %(vl_controle, vl_controle /(vl_controle + vl_paciente)*100))
    print('\n Valores faltantes: ', df.isnull().values.any())

    print('\nCaracteristicas com valores = 0 alteradas para media:', dt_feature.head())



# *********** preparando o modelo de ML ****************************

#criar uma lista de armazenamento de acuracia
accuracy_PC = []

#vetor beisiano Naive Bayes
accuracy_NB = []

def split_model():
    print('\n********************* resultado *************** \n')
    rould = 0.10
    epochs = 1
    for i in range(5):
        x_train, x_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3, random_state=1)
        print('divisao do conjunto de dados\n')
        print('x_train: %d\n y_train %d\n x_test %d\n y_test %d\n' %(len(x_train), len(y_train), len(x_test), len(y_test)))
        print('quantidade de amostras da classe 0: ', len(y_train.loc[y_train == 0]))
        print('quantidade de amostras da classe 1: ', len(y_train.loc[y_train == 1]))

        # Perceptron
        #percep = Perceptron(random_state=i)
        percep = Perceptron()
        percep.fit(x_train, y_train) #treinar em cima do conjunto de treinamento
        percep.predictions = percep.predict(x_test) # testar pra mim
        acc_percep = percep.score(x_test, y_test) # apresentar o resultado

        # Naive Bayes
        gnb = GaussianNB() #criado o classificador
        gnb.fit(x_train, y_train) # treinar o classificador
        gnb.predictions = gnb.predict(x_test) #testar o classificador com o conjunto de test
        acc_nb = gnb.score(x_test, y_test) # apresentar o resultado

        # Accuracy
        accuracy_PC.append(acc_percep)
        accuracy_NB.append(acc_nb)

        print('\n Resultados Perceptron: \n Acc_Perceptron: ', acc_percep)
        print('\n Resultados NB: \n Acc_Perceptron: ', acc_nb)
        print(metrics.confusion_matrix(y_test, percep.predictions))
        print('\n Classificacao: \n', metrics.classification_report(y_test, percep.predictions))

        print('\n Vetor de acuuracia Peceptron: ', accuracy_PC)
        print('\n Vetor de acuuracia NB: ', accuracy_NB)

        median = np.mean(accuracy_PC)
        print('vetor accuracy_PC - Media: ', median)




#****************** chamadas *****************************************
#information()

#target_count()
#correlation()
#split_model()

