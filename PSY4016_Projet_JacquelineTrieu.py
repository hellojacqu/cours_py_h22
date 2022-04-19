#!/usr/bin/env python
# coding: utf-8

# In[266]:


#importer les données sous la forme Dataframe
import pandas 
import numpy as np

fichier='/Users/jacquelinetrieu/Desktop/JacquelineNguyen_données_psy4016-H22_20220211_risk_factors_cervical_cancer.csv'
df_cervical=pandas.read_csv(fichier)
df_cervical


# In[267]:


#créer un nouveau dataframe avec les colonnes d'intérêts
df_cervical1=pandas.read_csv(fichier,
usecols = ['STDs (number)','Age', 'Number of sexual partners','Num of pregnancies','Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)'])
df_cervical1


# In[268]:


#renommer les colonnes selon bon format
df_cervicalnew= df_cervical1.rename(columns={'STDs (number)': 'STDs_number', 'Number of sexual partners': 'Number_sexualpartners','Num of pregnancies':'Num_pregnancies','Smokes (packs/year)':'Smokes_year','Hormonal Contraceptives (years)':'Hormonal_years','IUD (years)':'IUD_year'})


# In[275]:


#gestion et correction des erreurs, transformer point interrogation en chiffre
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Number_sexualpartners':
            if df_cervicalnew.at[x,c] == "?":
                df_cervicalnew.at[x,c] = "1000"

for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Number_sexualpartners':
            print(df_cervicalnew.at[x,c])
            
     


# In[276]:


#correction age, enlever la lettre

for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == "Age":
            if df_cervicalnew.at[x,c]== "26y": #je corrige les erreurs de la base de données
                df_cervicalnew.at[x,c] = "26"
            if df_cervicalnew.at[x,c]== "21y":
                df_cervicalnew.at[x,c] = "21"

for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Age':
            print(df_cervicalnew.at[x,c])
            


# In[277]:


#correction des erreurs pour les autres variables
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'STDs_number':
            if df_cervicalnew.at[x,c] == "?":
                df_cervicalnew.at[x,c] = "1000"         
                


# In[278]:


#gestion et correction des erreurs
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Num_pregnancies':
            if df_cervicalnew.at[x,c] == "?":
                df_cervicalnew.at[x,c] = "1000"


# In[279]:


#correction des erreurs
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Smokes_year':
            if df_cervicalnew.at[x,c] == "?":
                df_cervicalnew.at[x,c] = "1000"


# In[282]:


#correction des erreurs
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Hormonal_years':
            if df_cervicalnew.at[x,c] == "?":
                df_cervicalnew.at[x,c] = "1000"


# In[283]:


#correction des erreurs
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'IUD_year':
            if df_cervicalnew.at[x,c] == "?":
                df_cervicalnew.at[x,c] = "1000"


# In[285]:


#changer en entier et float

df_cervicalnew['Number_sexualpartners'] = df_cervicalnew['Number_sexualpartners'].astype(float)  
df_cervicalnew['Age'] = df_cervicalnew['Age'].astype(int)   
df_cervicalnew['STDs_number'] = df_cervicalnew['STDs_number'].astype(float)  
df_cervicalnew['Num_pregnancies'] = df_cervicalnew['Num_pregnancies'].astype(float)   
df_cervicalnew['Smokes_year'] = df_cervicalnew['Smokes_year'].astype(float)  
df_cervicalnew['Hormonal_years'] = df_cervicalnew['Hormonal_years'].astype(float)   


# In[288]:


#correction des erreurs
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Number_sexualpartners':
            if df_cervicalnew.at[x,c] == 1000:
                df_cervicalnew.at[x,c] = None


# In[289]:


#exemple de gestion erreur avec algorythme automatisation
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'STDs_number':
            if df_cervicalnew.at[x,c] == 1000:
                df_cervicalnew.at[x,c] = None
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Num_pregnancies':
            if df_cervicalnew.at[x,c] == 1000:
                df_cervicalnew.at[x,c] = None                
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Smokes_year':
            if df_cervicalnew.at[x,c] == 1000:
                df_cervicalnew.at[x,c] = None
for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Hormonal_years':
            if df_cervicalnew.at[x,c] == 1000:
                df_cervicalnew.at[x,c] = None                   


# In[ ]:





# In[290]:


for c in df_cervicalnew.columns:
    for x in df_cervicalnew.index:
        if c == 'Hormonal_years':
            print(df_cervicalnew.at[x,c])
            


# In[291]:


df_cervicalnew.values[30:39]


# In[293]:


#remplacer les valeurs manquantes par la valeur la plus fréquente
import sklearn 
from sklearn import impute
import numpy as np
import pandas
X_data = df_cervicalnew[["Number_sexualpartners","STDs_number","Num_pregnancies","Smokes_year","Hormonal_years"]]

imp = sklearn.impute.SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
imp.fit(X_data)
X_new = imp.transform(X_data)

df_cervicalnew[["Number_sexualpartners","STDs_number","Num_pregnancies","Smokes_year","Hormonal_years"]]=X_new

df_cervicalnew.values[10:39]


# In[258]:


#vérifier si présence valeurs manquantes
df_cervicalnew.isnull().sum()


# In[296]:


#code formatage de chaîne
df=df_cervicalnew[['STDs_number','Number_sexualpartners','Num_pregnancies', 'Smokes_year','Hormonal_years','IUD_year']].head()
#print(df)


for donnee in df:
    a=df[donnee]    
    var1 = a[0]
    var2 = a[1]
    var3 = a[2]
    var4 = a[3]
    print(f"{var1:4}", f"{var2:4}", f"{var3:4}", f"{var4:>4}")


# In[ ]:


STDs_number = input("colon avec les valeurs pour l\'age")
Education = input("collone avec les valeurs pour l\'education")


# In[297]:



import statsmodels
from statsmodels.formula.api import ols 


# In[263]:


#régression multiple objectif 1, influence des variables sur le nombre de STDs
model = ols("STDs_number ~ Age + Number_sexualpartners + Num_pregnancies + Smokes_year + Hormonal_years", df_cervicalnew).fit()

print(model.summary())


# In[298]:


#Graphiques objectif1
import seaborn as sns
sns.set_theme()


# In[302]:


g = sns.lmplot(
    data=df_cervicalnew,
    x="Age", y="STDs_number",
    height=5
)

g = sns.lmplot(
    data=df_cervicalnew,
    x="Number_sexualpartners", y="STDs_number",
    height=5
)


g = sns.lmplot(
    data=df_cervicalnew,
    x="Num_pregnancies", y="STDs_number",
    height=5
)


# In[414]:


#objectif 2

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


# In[415]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[416]:


contigency= df_cervical.crosstab(df_cervical['Biopsy'], df_cervical['Hinselmann'])
contigency


# In[417]:


c, p, dof, expected = chi2_contingency(contigency)
p


# In[418]:


#graphique tableau chi carré
plt.figure(figsize=(12,8))
sns.heatmap(contigency, annot=True, cmap="YlGnBu")


# In[419]:


# plan B: fisher test

data_crosstab = df_cervical.crosstab(data['Biopsy'],
                            data['Hinselmann'], 
                               margins = False)


# In[420]:


odd_ratio, p_value = stats.fisher_exact(data_crosstab)
print('odd ratio is : ' + str(odd_ratio))
print('p_value is : ' + str(p_value))


# In[334]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV


# In[305]:


#AA SUPERVISÉ

# je vais prendre les variables en lien avec STDs
X = df_cervicalnew2.loc[:, ["Age","Number_sexualpartners","Num_pregnancies","Smokes_year","Hormonal_years"]].astype(np.float64)
Y = df_cervicalnew2.loc[:, "STDs_number"].astype(np.float64)


# In[309]:


#ici, je normalise les données (pipeline format)
standard = StandardScaler()
normal = scaler.fit(X) 


# In[311]:


X = scaler.transform(X)


# In[314]:


from sklearn.model_selection import train_test_split
# je sépare les données en entrainement et en test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=50)


# In[315]:


X_train.shape


# In[316]:


X_test.shape


# In[317]:


y_train.shape


# In[318]:


y_test.shape


# In[330]:


# appliquer une régression logistique sur les données
analyse = LogisticRegression() 


# In[335]:


# faire une version de type cross-validation
analyse2= LogisticRegressionCV()


# In[337]:


# effectuer un entrainement
analyse.fit(X_train, y_train) 


# In[338]:


# tentative de cross validation training
analyse2.fit(X_train, y_train) 


# In[321]:


# ici je fais une prédiction sur les données de test
prediction = analyse.predict(X_test) 


# In[223]:


predicted


# In[405]:


plt.scatter(analyse.predict(X_train), analyse.predict(X_train)-y_train, c = 'b', s=40, alpha=0.5)
plt.scatter(analyse.predict(X_test), analyse.predict(X_test)-y_test, c='r', s=40)
plt.hlines(y=0, xmin=0, xmax=50)


# In[326]:


# ici on affiche la précision des prédictions
from sklearn.metrics import accuracy_score
precision = accuracy_score(y_test, prediction)
precision


# In[232]:



import seaborn as sns
sns.barplot(x = df_cervicalnew2["Age"], y = df_cervicalnew2["Number_sexualpartners"])


# In[352]:


#script SQLITE

import sqlite3
#on crée base de donnée sqlite

conn = sqlite3.connect('df_cervicalnew2.db') 
create_table = "CREATE TABLE IF NOT EXISTS data (Age INTEGER, Number_sexualpartners REAL, Num_pregnancies REAL, STDs_number REAL)"
cursor = conn.cursor()
cursor.execute(create_table)


# In[355]:


# ici je parcoure les donnees du dataframe 
#pour chaque ligne, j'insère les donnees désirée dans le nouveau cadre
for index, row in df_cervicalnew2.iterrows():
    a, b, c = row["Age"], row["Number_sexualpartners"], row["Num_pregnancies"]
    insert_query = "INSERT INTO data (Number_sexualpartners, Age, Num_pregnancies) VALUES                 (?, ?, ?)"
    cursor.execute(insert_query, (a, b, c))


# In[406]:


#APPRENTISSAGE NON SUPERVISÉ CODE
from sklearn.cluster import KMeans


dfcervical_kmeans = df_cervicalnew2.drop(['STDs_number', 'Age', 'Number_sexualpartners', 'Num_pregnancies'],axis=1)

# kmean marche avec valeur continue. donc 
#on supprime toutes les valeurs categoriels puisque l'algorithme k-means marche pas avec. il utilise des 

print(dfcervical_kmeans.columns)
print(dfcervical_kmeans.dtypes)
pattern = r"^\d*$" # expression reguliere qui represente un entier

print(dfcervical_kmeans['Smokes_year'][0:190])
    
print(dfcervical_kmeans.head())


# In[421]:


# ceci est une fonction anonyme qui convertit toutes les valeurs en entier
dfcervical_kmeans["Smokes_year"] = dfcervical_kmeans["Smokes_year"].apply(lambda x: int(x))


# In[422]:


# classe permettant de faire de l'apprentissage non supervise avec l'algorithme kmeans
class NonSup:
    def __init__(self, data):
        self.data = data   # donnees d'entrainements
        self.centroids = None
        
    def entrainer(self):
        kmeans_algo = KMeans(n_clusters=5)
        kmeans_algo.fit(self.data)
        self.centroids = kmeans_algo.cluster_centers_
        
    def prepare(self):
        self.data = self.data.astype(np.float32)
        scaler = StandardScaler()
        scaled = scaler.fit(self.data) # normaliser les donnees
        self.data= scaler.transform(self.data)
        print(self.data.shape)
        
    def afficher(self):
        labels = kmeans_algo.predict(self.data)
        print(labels)
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=80, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='yellow', s=500, alpha=0.8);
        plt.show()


# In[423]:


AANonSuper = NonSupervise(dfcervical_kmeans[["Smokes_year", "Hormonal_years"]])


# In[424]:


AANonSuper.prepare()


# In[425]:


AANonSuper.entrainer()


# In[428]:


AANonSuper = NonSupervise(dfcervical_kmeans[["Smokes_year", "Hormonal_years"]])
AANonSuper.prepare()
AANonSuper.entrainer()
AANonSuper.afficher() #graphique ne s'affiche pas ,mais on le fait avec commande afficher


# In[393]:





# In[394]:


import sklearn

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
print('sklearn version:', sklearn.__version__)


# In[395]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[396]:


from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                    cluster_std=0.6, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);


# In[397]:


from sklearn.cluster import KMeans


# In[399]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# In[400]:


plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[ ]:


# Ajuster aux données et prédire à l'aide de GNB et l'ACP en pipeline.
# pour les données non mises à l'échelle

#exemple de pipeline
from sklearn import pipeline
from sklearn import decomposition
from sklearn import naive_bayes

# unscaled_clf = sklearn.pipeline.make_pipeline(decomposition.PCA(n_components=2), naive_bayes.GaussianNB())
unscaled_clf = pipeline.make_pipeline(decomposition.PCA(n_components=2),
                                      naive_bayes.GaussianNB())

unscaled_clf.fit(X_train, y_train)
pred_test = unscaled_clf.predict(X_test)

# pour les données mises à l'échelle
# std_clf = sklearn.pipeline.make_pipeline(preprocessing.StandardScaler(), decomposition.PCA(n_components=2), naive_bayes.GaussianNB())
std_clf = pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                 decomposition.PCA(n_components=2),
                                 naive_bayes.GaussianNB())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)

