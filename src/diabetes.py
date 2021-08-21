import os
import pandas as pd
import numpy as np

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
print(dt_feature)
print(dt_target)
#print('\n ************ Tratando dados iuais a 0 *************')
#print('colunas de valores igual a 0:\n', (df==0).sum())

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

information()

