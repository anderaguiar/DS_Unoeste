import os
import pandas as pd

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
print('\n **************** Informações sobre o DataSet *************** \n')
print('Diretorios: \n')
print('Meu diretorio do projeto: \n', BASE_DIR)
print('Meu diretorio de dados: \n', DATA_DIR)
print('Este é meu dataset: \n', df.head(5))
