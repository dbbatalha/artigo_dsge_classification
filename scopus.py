
import unicodedata
import re
import spacy
import scipy
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas.core.arrays import integer
from pandas.core.frame import DataFrame
from pybliometrics.scopus import ScopusSearch , CitationOverview , AbstractRetrieval

from spacy.lang.en.stop_words import STOP_WORDS as stopwords

from sklearn.model_selection import GroupKFold , RandomizedSearchCV , KFold , cross_validate, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint





nlp = spacy.load('en_core_web_sm')

# s = ScopusSearch("DSGE")

# print(len(s.results))

resultados = s.results

resultados[0]


df = pd.DataFrame(resultados)


drop_list = ['pubmed_id','doi','pii','subtype', 'affiliation_city','author_afids','eIssn','aggregationType','volume','pageRange','coverDisplayDate','article_number','issueIdentifier','openaccess','fund_no','fund_acr','fund_sponsor']
for i in drop_list:
    df.drop(i, axis = 1, inplace=True )
# df.columns

df.rename(columns={"title": "titulo", "subtypeDescription": "tipo", "creator": "autor", "afid": "instituto_id", "affilname": "instituto_nome", "affiliation_country": "pais", "author_count": "num_autores", "author_names": "autores", "author_ids": "autor_ids", "coverDate": "data", "description": "descricao", "authkeywords": "tags", "citedby_count": "num_citacoes", 'publicationName':'veiculo'}, inplace=True)

df.sort_values(by=['num_citacoes'], ascending=False)

df.to_pickle("dsge_abstract.pkl") #salvando meu dataframe
df = pickle.load(open("dsge_abstract.pkl", "rb"))
# df.to_csv('dsge_abstract.csv', index=False)



# ---------

#functions for process
def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x


def make_to_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)


def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x


#Descri_ process


df['num_palavras'] = df['descricao'].apply(lambda x: len(str(x).split()))

df['ano']=df['data'].apply(lambda x: str(x)[:4])
df.drop('data', axis =1)


#contracoes

contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how does",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    # "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    " u ": " you ",
    " ur ": " your ",
    " n ": " and ",
    "won't": "would not",
    'dis': 'this',
    'bak': 'back',
    'brng': 'bring'
}

# -----

df['descricao_proc'] = df['descricao'].apply(lambda x: cont_to_exp(x))
df.dtypes
#removendo chracter special
df['descricao_proc'] = df['descricao_proc'].apply(lambda x: re.sub(r'[^\w ]+', "", x))


df['tags_proc'] = df['tags']


# Process Descri_ e Tags_


Colunas = ['tipo','autor', 'veiculo', 'instituto_nome','descricao_proc', 'tags_proc', 'pais']
for i in Colunas:
    #deixando tudo em lower case
    df[i] = df[i].apply(lambda x: str(x).lower())

    #removendo espacos multiplos
    df[i] = df[i].apply(lambda x: ' '.join(x.split())) 
    #esse joint irá juntar as palavras com espaco simples

    #removendo os acentos
    df[i] = df[i].apply(lambda x: remove_accented_chars(x))

    #removendo as stopwords
    df[i] = df[i].apply(lambda x: ' '.join([t for t in x.split() if t not in stopwords]))

    #transformando na raiz da palavra
    df[i] = df[i].apply(lambda x: make_to_base(x))

df.sort_values(by=['num_citacoes'], ascending=False).head(30)




#-----------

#EDA

dsge_filter = df.loc[df['descricao_proc'].str.contains('monetary') & df['ano'].str.startswith('20')] # Filtrando por 2 parametros
len (dsge_filter)

veiculos=['economic modelling', 'journal economic dynamic control','journal money , credit banking','journal macroeconomic','journal monetary economic','macroeconomic dynamic','international journal central banking' ,'journal international money finance' ,'european economic review' ,'economic letter']


dsge_filter = dsge_filter[dsge_filter.veiculo.isin(veiculos)]

dsge_filter.head()






#melhores selecoes para a quantidade de artigos





economy=['','economy', 'policy', 'macroeconomics']
for j in economy:
    k = len(dsge_filter.loc[dsge_filter['tags_proc'].str.contains(j)])
    print('Tamanho de {} é {}.'.format(j, k))

dsge_filter.tags_proc.unique()








dgse_filter = dsge_filter.sort_values(by=['num_citacoes'], ascending=False)
dgse_filter

#tipos do arquivo

dsge_filter.tipo.value_counts()





#valores unicos de tags
lista_de_tags = []
for tags in dsge_filter.Tags_proc.unique():
    for tag in tags.split('|'):
        tag =  ' '.join(tag.split())
        if tag not in lista_de_tags:
            lista_de_tags.append(tag)
        
print(lista_de_tags)
len(lista_de_tags)



#valores unicos veiculo
dsge_filter.veiculo.unique()
len(dsge_filter.veiculo.unique())

dsge_filter['veiculo'].value_counts().head(25)



#valores unicos institutos
dsge_filter.instituto_nome.unique()
len(dsge_filter.instituto_nome.unique())


dsge_filter['instituto_nome'].value_counts()



def inst_count(dataframe):
    lista_inst = []
    for institutos in dataframe.instituto_nome.unique():
        for instituto in institutos.split(';'):
            if instituto not in lista_inst:
                if instituto !='':
                    lista_inst.append(instituto)

    lista_quant_inst = []        
    for instituto in lista_inst:
        quantidade_inst = []
        for linha_inst in dataframe.instituto_nome:
            if instituto in linha_inst:
                quantidade_inst.append(1)
            else:
                quantidade_inst.append(0)
        tamanho = sum(quantidade_inst)
        lista_quant_inst.append(tamanho)
    total_inst = list(zip(lista_inst, lista_quant_inst))
    return total_inst


ls3 = pd.DataFrame(inst_count(dsge_filter), columns=['instituto', 'quantidade']).sort_values(by=['quantidade'], ascending=False)[:20]



sns.barplot(x="instituto", y="quantidade", data=ls3)
plt.xlabel("Instituto", fontsize=15)
plt.ylabel("Número de Publicações", fontsize=15)
plt.xticks(rotation=80)

plt.show()



#valores unicos por autor
dsge_filter.autor.unique()
len(dsge_filter.autor.unique())

dsge_filter['autor'].value_counts()



dgse_filter1 = dsge_filter.sort_values(by=['num_citacoes'], ascending=False)[:20]

sns.barplot(x="autor", y="num_citacoes", data=dgse_filter1)
plt.xlabel("Autor", fontsize=15)
plt.ylabel("Número de citações", fontsize=15)
plt.xticks(rotation=80)

plt.show()


#criando a coluna de tags com apenas 3 variaveis
lista_tags1=[]
for tags in dsge_filter.Tags_proc:
    lista_1 = []
    if len(tags.split('|')) >= 3:
        for tag in tags.split('|')[:2]:
            tag =  ' '.join(tag.split())
            lista_1.append(tag)
    else:
        tag =  ' '.join(tag.split())
        lista_1.append(tag)
    lista_tags1.append(lista_1)

lista_tags1




#contagem de países

def country_count(dataframe):
    lista_paises = []
    for paises in dataframe.pais.unique():
        for pais in paises.split(';'):
            if pais not in lista_paises:
                if pais !='':
                    lista_paises.append(pais)

    lista_quant_paises = []        
    for pais in lista_paises:
        quantidade_pais = []
        for linha_pais in dataframe.pais:
            if pais in linha_pais:
                quantidade_pais.append(1)
            else:
                quantidade_pais.append(0)
        tamanho = sum(quantidade_pais)
        lista_quant_paises.append(tamanho)
    total_pais = list(zip(lista_paises, lista_quant_paises))
    return total_pais

paises = pd.DataFrame(country_count(dsge_filter), columns=['pais','quantidade']).sort_values('quantidade', ascending=False)[1:20]


sns.barplot(x="pais", y="quantidade", data=paises)
plt.xlabel("País", fontsize=15)
plt.ylabel("Número de publicações", fontsize=15)
plt.xticks(rotation=80)

plt.show()





## Machine Learning


a_trocar={'economic modelling' : 0,
    'journal economic dynamic control':1,
    'journal money , credit banking': 2,
    'journal macroeconomic' : 3,
    'journal monetary economic': 4,
    'macroeconomic dynamic': 5,
    'international journal central banking': 6,
    'journal international money finance' : 7,
    'european economic review' : 8,
    'economic letter' : 9
}

dsge_filter.veiculo = dsge_filter.veiculo.map(a_trocar)
dsge_filter.head()
len(dsge_filter)

#arvore de decisão










def imprime_score(scores):
  media = scores.mean() * 100
  desvio = scores.std() * 100
  print("Accuracy médio %.2f" % media)
  print("Intervalo [%.2f, %.2f]" % (media - 2 * desvio, media + 2 * desvio))







tfidf = TfidfVectorizer()
x = tfidf.fit_transform(dsge_filter['descricao_proc'])

y = dsge_filter['veiculo']

SEED = 301
np.random.seed(SEED)

modelo = DummyClassifier()
results = cross_validate(modelo, x, y, cv = 10, return_train_score=False)
media = results['test_score'].mean()
desvio_padrao = results['test_score'].std()
print("Accuracy com dummy stratified, 10 = [%.2f, %.2f]" % ((media - 2 * desvio_padrao)*100, (media + 2 * desvio_padrao) * 100))


#Cross Validation com RandomizedSearchCV




SEED=301
np.random.seed(SEED)

espaco_de_parametros = {
    "max_depth" : [10, 15, 20, 30, 40 ],
    "min_samples_split" : [4, 8, 16, 32, 64, 128],
    "min_samples_leaf" : [2, 4, 8, 16],
    "criterion" : ["gini", "entropy"]
}

busca = RandomizedSearchCV(DecisionTreeClassifier(),
                    espaco_de_parametros,
                    n_iter = 32, # esse aqui é a quantidade de grupo de parametros que selecionarei dentre os 36 elementos (2*3*3*2)
                    cv = KFold(n_splits = 10, shuffle=True),
                    random_state = SEED)
busca.fit(x, y)
resultados = pd.DataFrame(busca.cv_results_)
resultados.head()

resultados_ordenados_pela_media = resultados.sort_values("mean_test_score", ascending=False)
for indice, linha in resultados_ordenados_pela_media.iterrows():
  print("%.3f +-(%.3f) %s" % (linha.mean_test_score, linha.std_test_score*2, linha.params))

scores = cross_val_score(busca, x, y, cv = KFold(n_splits=5, shuffle=True))
imprime_score(scores)
melhor = busca.best_estimator_
print(melhor)



#Random Forest


SEED=301
np.random.seed(SEED)

espaco_de_parametros = {
    "n_estimators" : [10, 50, 100],
    "max_depth" : [3, 5, 10, 15, 20, 30],
    "min_samples_split" : [4, 8, 16, 32, 64, 128],
    "min_samples_leaf" : [4, 8, 16, 32, 64, 128],
    "bootstrap" : [True, False],
    "criterion" : ["gini", "entropy"]
}

busca = RandomizedSearchCV(RandomForestClassifier(),
                    espaco_de_parametros,
                    n_iter = 32,
                    cv = KFold(n_splits = 5, shuffle=True))
busca.fit(x, y)

resultados = pd.DataFrame(busca.cv_results_)
resultados.head()

resultados_ordenados_pela_media = resultados.sort_values("mean_test_score", ascending=False)
for indice, linha in resultados_ordenados_pela_media[:5].iterrows():
  print("%.3f +-(%.3f) %s" % (linha.mean_test_score, linha.std_test_score*2, linha.params))



scores = cross_val_score(busca, x, y, cv = KFold(n_splits=5, shuffle=True))
imprime_score(scores)

melhor = busca.best_estimator_
print(melhor)























































