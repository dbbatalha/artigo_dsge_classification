import matplotlib
import pandas as pd
from pandas.core.arrays import integer
from pandas.core.frame import DataFrame
from pybliometrics.scopus import CitationOverview , AbstractRetrieval
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from pybliometrics.scopus import ScopusSearch
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
import unicodedata
import re
import spacy

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




#df.to_pickle("dsge_abstract.pkl") #salvando meu dataframe

df = pd.read_csv('dsge_abstract.csv')
df.sort_values(by=['num_citacoes'], ascending=False)
df.head()

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

dsge_filter = df.loc[df['descricao_proc'].str.contains('monetary') &  df['ano'].str.startswith('20')] # Filtrando por 2 parametros
len (dsge_filter)


#melhores selecoes para a quantidade de artigos
economy=['','economy', 'policy', 'macroeconomics']
for j in economy:
    k = len(dsge_filter.loc[dsge_filter['tags_proc'].str.contains(j)])
    print('Tamanho de {} é {}.'.format(j, k))






dgse_filter = dsge_filter.sort_values(by=['num_citacoes'], ascending=False)
dgse_filter





#valores unicos de tags
lista_de_tags = []
for tags in dsge_filter.Tags_proc.unique():
    for tag in tags.split('|'):
        tag =  ' '.join(tag.split())
        if tag not in lista_de_tags:
            lista_de_tags.append(tag)
        
print(lista_de_tags)
len(lista_de_tags)

