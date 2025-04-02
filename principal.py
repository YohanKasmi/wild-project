######################## IMPORTS #########################################

import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import base64

#ML
from sklearn.preprocessing import StandardScaler  #avec écart-type
from sklearn.preprocessing import MinMaxScaler    #sensible aux valeurs extrêmes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

########################### AFFICHAGE EN MODE WIDE (LARGE) ##################
st.set_page_config(layout="wide")

######################## IMPORTS CSV ####################################

df_titres = pd.read_csv('df_titres.csv')
df_names = pd.read_csv('df_names.csv')

df_acteurs = pd.read_csv('df_acteurs.csv')
df_realisateurs = pd.read_csv('df_realisateurs.csv')
df_liste_noms = pd.read_csv('df_liste_noms.csv')
df_fr = pd.read_csv('df_fr.csv')
df_etranger = pd.read_csv('df_etranger.csv')
df_genres = pd.read_csv('df_genres.csv')
df_genre_act = pd.read_csv('df_genre_act.csv')
df_genre_real = pd.read_csv('df_genre_real.csv')

######################## CREATION D'UNE LISTE QUI NOUS SERVIRA POUR LE TRI DES FILMS PAR GENRE ET LA SELECTBOX ####################################

liste_genres = ['Action', 'Actualités', 'Animation', 'Aventure', 'Biographie',
       'Comédie', 'Crime', 'Documentaire', 'Drame',
       'Famille', 'Fantastique', 'Guerre', 'Histoire', 'Horreur', 'Musique',
       'Mystère', 'Romance', 'Science-Fiction', 'Sport', 'Thriller',
       'Téléfilm', 'Western']

######################## PREPARATION DES DONNEES EN LISTE ####################################

col_a_modif = ['liste_realisateurs', 'liste_acteurs', 'imdb_suggestion','tconst_suggestion'] 
for col in col_a_modif :
  df_titres[col]=df_titres[col].str.split(',')

col_a_modif2 = ['commeActeur', 'commeRealisateur']
for col in col_a_modif2 :
  df_names[col]=df_names[col].str.split(',')



######################## Définition des fonctions ###############################

# POUR OBTENIR L'ID DU FILM A PARTIR DE SON TITRE ----------------------------------------
def id_du_film(titre) : # renvoie le tconst d'un film à partir du titreVF
  if  titre in df_titres['titreVF'].tolist() :  # Vérification que le film est bien connu                                  
    return df_titres.loc[df_titres['titreVF'] == titre]['tconst'].iloc[0]
  elif  titre in df_titres['TitreOriginal'].tolist() :  # Vérification que le film est bien connu                                  
    return df_titres.loc[df_titres['TitreOriginal'] == titre]['tconst'].iloc[0]   
  else : 
     st.write("Ce film n'est pas connu, vérifiez l'orthographe") #Si le film n'est pas trouvé dans notre BDD


# ON TRI LE DATAFRAME POUR GARDER QUE LES FILMS POUVANT CORRESPONDRE ----------------------------------------
def df_tri(film) :   # renvoie le dataframe utile pour les voisins en fonction du tconst
  # Création et ajustement du modèle NearestNeighbors
  df_act = df_titres # Création d'un dataframe de travail pour cette fonction
  
  langue = df_act.loc[df_act['tconst'] == film]['original_language'].iloc[0] # On récupère la langue du film
  df_act = df_act.loc[df_act['original_language'] == langue][['tconst','liste_acteurs','liste_realisateurs','anneeSortie','noteMoyenne', 'nbVotes']] # récupération des colonnes intéressantes pour les films dans la langue
  df_act['act_commun'] = 0 # Initialisation d'une liste pour compter les acteurs en commun
  df_act['real_commun'] = 0 # Initialisation d'une liste pour compter les réalisateurs en commun
  if df_act.loc[df_act['tconst']==film]['liste_acteurs'].iloc[0][0] != 'pas_acteurs' :
    for act in df_act.loc[df_act['tconst']==film]['liste_acteurs'].iloc[0] : # On parcourt la liste des acteurs du film choisi
    
      for film_act in df_names.loc[df_names['nconst']== act]['commeActeur'].iloc[0] : # On parcourt la liste des films des acteurs
        df_act.loc[df_act['tconst'] == film_act, 'act_commun'] += 1 # Incrémentation du nombre d'acteur en commun
 
  for real in df_act.loc[df_act['tconst']==film]['liste_realisateurs'].iloc[0] : 
    for film_real in df_names.loc[df_names['nconst']== real]['commeRealisateur'].iloc[0] :
        df_act.loc[df_act['tconst'] == film_real, 'real_commun'] += 1
  df_act = df_act[['tconst','anneeSortie','noteMoyenne', 'nbVotes', 'act_commun', 'real_commun']]
  
   #Normalisation des données :
  scaler = StandardScaler()           # J'utilise StandardScaler pour qu'il soit moins sensible aux valeurs extrêmes
  scaled = scaler.fit_transform(df_act[['anneeSortie','noteMoyenne', 'nbVotes','act_commun', 'real_commun']])
  df_scaled = pd.DataFrame(scaled, columns=['anneeSortie','noteMoyenne', 'nbVotes','act_commun', 'real_commun'])
  df_scaled['act_commun'] = df_scaled['act_commun']/4 # On ajuste le poids des différentes colonnes
  df_scaled['real_commun'] = df_scaled['real_commun']/5
  df_scaled['noteMoyenne'] = df_scaled['noteMoyenne']*6
  df_scaled['anneeSortie'] = df_scaled['anneeSortie']*3
  df_scaled['nbVotes'] = df_scaled['nbVotes']*3
  
  df_act = pd.concat([df_act['tconst'].reset_index(drop=True), df_scaled], axis=1)
  df = pd.merge(df_act,            #création d'un df avec toutes les données pour le KNN
               df_genres,
               how = 'inner',
               on = 'tconst')
  df = df.drop(['Unnamed: 0'], axis = 1)   
  
  # tri sur les genres : suppression des films qui n'ont aucun genre en commun
  #obtenir la liste des genres du film :
  genres_film = [genre for genre in liste_genres if df.loc[df['tconst']==film][genre].iloc[0] == 1]
  #on supprime tous les films qui n'ont pas de genre en commun et de genre important
  liste_genre_princ = ['Animation', 'Horreur','Comédie', 'Crime', 'Documentaire', 'Famille', 'Guerre', 'Western', 'Science-Fiction','Histoire']
  if len(genres_film) == 1 :                    # si le film n'a qu'un genre, on garde uniquement les films qui contiennent ce genre
    df = df.loc[~(df[genres_film[0]] == 0)]
  
  elif len(genres_film) == 2 :
    df = df.loc[~((df[genres_film[0]] == 0) & (df[genres_film[1]] == 0))]     # si le film a deux genres, on supprime tous les films qui n'ont aucun de ces deux genres
    if genres_film[0] in liste_genre_princ :          # si le film a un genre parmis les genres principaux, on supprime tous les films n'ayant pas ce genre principal 
         df = df.loc[~(df[genres_film[0]] == 0)]
    if genres_film[1] in liste_genre_princ :
         df = df.loc[~(df[genres_film[1]] == 0)]   
  elif len(genres_film) == 3 :
    df = df.loc[~((df[genres_film[0]] == 0) & (df[genres_film[1]] == 0) & (df[genres_film[2]] == 0))]   # si le film a 3 genres, on supprime tous les films qui n'ont aucun de ces 3 genres
    if genres_film[0] in liste_genre_princ :         # si le film a un genre parmis les genres principaux, on supprime tous les films n'ayant pas ce genre principal 
        df = df.loc[~(df[genres_film[0]] == 0)]
    if genres_film[1] in liste_genre_princ :
        df = df.loc[~(df[genres_film[1]] == 0)]
    if genres_film[2] in liste_genre_princ :
        df = df.loc[~(df[genres_film[2]] == 0)]
  elif len(genres_film) == 4 :
    df = df.loc[~((df[genres_film[0]] == 0) & (df[genres_film[1]] == 0) & (df[genres_film[2]] == 0)  & (df[genres_film[3]] == 0))]   # si le film a 4 genres, on supprime tous les films qui n'ont aucun de ces 3 genres
    if genres_film[0] in liste_genre_princ :         # si le film a un genre parmis les genres principaux, on supprime tous les films n'ayant pas ce genre principal 
        df = df.loc[~(df[genres_film[0]] == 0)]
    if genres_film[1] in liste_genre_princ :
        df = df.loc[~(df[genres_film[1]] == 0)]
    if genres_film[2] in liste_genre_princ :
        df = df.loc[~(df[genres_film[2]] == 0)]
    if genres_film[3] in liste_genre_princ :
        df = df.loc[~(df[genres_film[3]] == 0)]
  elif len(genres_film) == 5 :
    df = df.loc[~((df[genres_film[0]] == 0) & (df[genres_film[1]] == 0) & (df[genres_film[2]] == 0)  & (df[genres_film[3]] == 0) & (df[genres_film[4]] == 0))]   # si le film a 4 genres, on supprime tous les films qui n'ont aucun de ces 3 genres
    if genres_film[0] in liste_genre_princ :         # si le film a un genre parmis les genres principaux, on supprime tous les films n'ayant pas ce genre principal 
        df = df.loc[~(df[genres_film[0]] == 0)]
    if genres_film[1] in liste_genre_princ :
        df = df.loc[~(df[genres_film[1]] == 0)]
    if genres_film[2] in liste_genre_princ :
        df = df.loc[~(df[genres_film[2]] == 0)]
    if genres_film[3] in liste_genre_princ :
        df = df.loc[~(df[genres_film[3]] == 0)]
    if genres_film[4] in liste_genre_princ :
        df = df.loc[~(df[genres_film[4]] == 0)]
  elif len(genres_film) == 6 :
    df = df.loc[~((df[genres_film[0]] == 0) & (df[genres_film[1]] == 0) & (df[genres_film[2]] == 0)  & (df[genres_film[3]] == 0) & (df[genres_film[4]] == 0))]   # si le film a 4 genres, on supprime tous les films qui n'ont aucun de ces 3 genres
    if genres_film[0] in liste_genre_princ :         # si le film a un genre parmis les genres principaux, on supprime tous les films n'ayant pas ce genre principal 
        df = df.loc[~(df[genres_film[0]] == 0)]
    if genres_film[1] in liste_genre_princ :
        df = df.loc[~(df[genres_film[1]] == 0)]
    if genres_film[2] in liste_genre_princ :
        df = df.loc[~(df[genres_film[2]] == 0)]
    if genres_film[3] in liste_genre_princ :
        df = df.loc[~(df[genres_film[3]] == 0)]
    if genres_film[4] in liste_genre_princ :
        df = df.loc[~(df[genres_film[4]] == 0)]
    if genres_film[5] in liste_genre_princ :
        df = df.loc[~(df[genres_film[5]] == 0)]
  elif len(genres_film) == 7 :
    df = df.loc[~((df[genres_film[0]] == 0) & (df[genres_film[1]] == 0) & (df[genres_film[2]] == 0)  & (df[genres_film[3]] == 0) & (df[genres_film[4]] == 0))]   # si le film a 4 genres, on supprime tous les films qui n'ont aucun de ces 3 genres
    if genres_film[0] in liste_genre_princ :         # si le film a un genre parmis les genres principaux, on supprime tous les films n'ayant pas ce genre principal 
        df = df.loc[~(df[genres_film[0]] == 0)]
    if genres_film[1] in liste_genre_princ :
        df = df.loc[~(df[genres_film[1]] == 0)]
    if genres_film[2] in liste_genre_princ :
        df = df.loc[~(df[genres_film[2]] == 0)]
    if genres_film[3] in liste_genre_princ :
        df = df.loc[~(df[genres_film[3]] == 0)]
    if genres_film[4] in liste_genre_princ :
        df = df.loc[~(df[genres_film[4]] == 0)]
    if genres_film[5] in liste_genre_princ :
        df = df.loc[~(df[genres_film[5]] == 0)]
    if genres_film[6] in liste_genre_princ :
        df = df.loc[~(df[genres_film[6]] == 0)]
  elif len(genres_film) == 8 :
    df = df.loc[~((df[genres_film[0]] == 0) & (df[genres_film[1]] == 0) & (df[genres_film[2]] == 0)  & (df[genres_film[3]] == 0) & (df[genres_film[4]] == 0))]   # si le film a 4 genres, on supprime tous les films qui n'ont aucun de ces 3 genres
    if genres_film[0] in liste_genre_princ :         # si le film a un genre parmis les genres principaux, on supprime tous les films n'ayant pas ce genre principal 
        df = df.loc[~(df[genres_film[0]] == 0)]
    if genres_film[1] in liste_genre_princ :
        df = df.loc[~(df[genres_film[1]] == 0)]
    if genres_film[2] in liste_genre_princ :
        df = df.loc[~(df[genres_film[2]] == 0)]
    if genres_film[3] in liste_genre_princ :
        df = df.loc[~(df[genres_film[3]] == 0)]
    if genres_film[4] in liste_genre_princ :
        df = df.loc[~(df[genres_film[4]] == 0)]
    if genres_film[5] in liste_genre_princ :
        df = df.loc[~(df[genres_film[5]] == 0)]
    if genres_film[6] in liste_genre_princ :
        df = df.loc[~(df[genres_film[6]] == 0)]
    if genres_film[7] in liste_genre_princ :
        df = df.loc[~(df[genres_film[7]] == 0)]
  elif len(genres_film) == 9 :
    df = df.loc[~((df[genres_film[0]] == 0) & (df[genres_film[1]] == 0) & (df[genres_film[2]] == 0)  & (df[genres_film[3]] == 0) & (df[genres_film[4]] == 0))]   # si le film a 4 genres, on supprime tous les films qui n'ont aucun de ces 3 genres
    if genres_film[0] in liste_genre_princ :         # si le film a un genre parmis les genres principaux, on supprime tous les films n'ayant pas ce genre principal 
        df = df.loc[~(df[genres_film[0]] == 0)]
    if genres_film[1] in liste_genre_princ :
        df = df.loc[~(df[genres_film[1]] == 0)]
    if genres_film[2] in liste_genre_princ :
        df = df.loc[~(df[genres_film[2]] == 0)]
    if genres_film[3] in liste_genre_princ :
        df = df.loc[~(df[genres_film[3]] == 0)]
    if genres_film[4] in liste_genre_princ :
        df = df.loc[~(df[genres_film[4]] == 0)]
    if genres_film[5] in liste_genre_princ :
        df = df.loc[~(df[genres_film[5]] == 0)]
    if genres_film[6] in liste_genre_princ :
        df = df.loc[~(df[genres_film[6]] == 0)]
    if genres_film[7] in liste_genre_princ :
        df = df.loc[~(df[genres_film[7]] == 0)]
    if genres_film[8] in liste_genre_princ :
        df = df.loc[~(df[genres_film[8]] == 0)]
        
  if 'Animation' not in genres_film :       # On ne suggère pas des films d'animation si le film de base n'est pas un film d'animation
     df = df.loc[~(df['Animation'] == 1)]
  if 'Horreur' not in genres_film :         # On ne suggère pas des films d'horreur si le film de base n'est pas un film d'horreur
     df = df.loc[~(df['Horreur'] == 1)]

  return df
  

# Algorithme de la sugestion des films seon le dataframe trié au dessus ----------------------------------------

def suggestions(df, film) :

  array = df.iloc[:,1:].to_numpy()           # transformation des valeurs utiles en array (j'aurais aussi pu utiliser un .values)
  nn = NearestNeighbors(n_neighbors=min(25, len(array)), metric='euclidean')         # pour récupérer 10 films voisins
  nn.fit(array)
  mon_film = df.loc[df['tconst']==film].iloc[:,1:].to_numpy()     
  distances, indices = nn.kneighbors(mon_film)     # on récupère les distances et les indices des films les plus proches

  liste_distances = distances.tolist()     # transformation de l'array des distances en liste
  liste_indices = indices.tolist()         # transformation de l'array des indices en liste

  del liste_indices[0][0]                 # supprime le premier de la liste (qui est le film cible)
  liste_indices = liste_indices[0]        # transforme la liste de liste en liste simple
  liste_finale = []
  liste_finale = [df_titres.loc[df_titres['tconst']==df.iloc[i]['tconst']][['titreVF']].iloc[0].iloc[0] for i in liste_indices] # La liste finale des titres
  liste_imdb = [df_titres.loc[df_titres['tconst']==df.iloc[i]['tconst']][['id']].iloc[0].iloc[0] for i in liste_indices] # la liste finale des affiches tmdb
  return liste_finale, liste_imdb


# Filtrage du dataframe selon l'acteur sélectionné ----------------------------------------

def filmograhie_acteur(nom_acteur) :
    
    IDactor = df_names.loc[df_names['Nom']==nom_acteur]['nconst'].iloc[0]  # recherche du nconst de l'acteur
    if df_names.loc[df_names['Nom']==nom_acteur]['commeActeur'].iloc[0] == 'pas_de_film' : # Si ce n'est pas un acteur retour d'une liste de film vide
       liste_finale = []
       liste_imdb = []
    else : # Si c'est un acteur on garde juste ses films du dataframe df_titres
      df_actor_choisi = df_acteurs.loc[df_acteurs['nconst'] == IDactor]['tconst'] # Sur le dataframe des acteurs on garde juste les lignes du nconst choisi
      df = pd.merge(df_titres,df_actor_choisi, how = 'inner', on = 'tconst') # On le merge avec le df_titres pour trouver les films correspondants
      df = df.drop_duplicates(subset= ['tconst'])
      df = df.sort_values('nbVotes', ascending = False).head(20) # On les classe par nombre de votes
      liste_finale = df['tconst'].tolist() # On transforme la colonne des titres en liste
      liste_tmdb = df['id'].tolist() # On transforme la colonne des id tmd en liste
      
    return liste_finale, liste_tmdb # On retourne les 2 listes
 

# Filtrage du dataframe selon le réalisateur sélectionné ----------------------------------------

def filmograhie_realisateur(nom_real) :
    IDactor = df_names.loc[df_names['Nom']==nom_real]['nconst'].iloc[0] # recherche du nconst du réalisateur
    if df_names.loc[df_names['Nom']==nom_real]['commeRealisateur'].iloc[0] == 'pas_de_film' : # Si ce n'est pas un réalisateur on retourne une liste de film vide
        liste_real = []
    else : # Si c'est un réalisateur on garde juste ses films du dataframe df_titres
         df_real_choisi = df_realisateurs.loc[df_realisateurs['nconst'] == IDactor]['tconst'] # Sur le dataframe des réalisateurs on garde juste les lignes du nconst choisi
         df = pd.merge(df_titres,df_real_choisi, how = 'inner', on = 'tconst') # On le merge avec le df_titres pour trouver les films correspondants
         df = df.sort_values('nbVotes', ascending = False).head(20) # On les classe par nombre de votes
         liste_finale = df['tconst'].tolist() # On transforme la colonne des titres en liste
         liste_tmdb  = df['id'].tolist() # On transforme la colonne des id tmd en liste
    return liste_finale, liste_tmdb # On retourne les 2 listes

# liste des acteurs de même genre
def meme_genre(nom, role) :
  
  nconst = df_names.loc[df_names['Nom']==nom]['nconst'].iloc[0]
  if role == 'acteur' :
    df = df_genre_act
  else :
    df = df_genre_real
  index_value = df.loc[df['nconst']==nconst].index

  scaler = StandardScaler()
  
  array = df.iloc[:,2:].to_numpy()           # transformation des valeurs utiles en array (sans prendre la 1ere colonne qui est un nconst)
  array = scaler.fit_transform(array)
  
  array[:,23] = array[:,23]
  array[:,24] = array[:,24]*10
  array[:,25] = array[:,25]/2
  nn = NearestNeighbors(n_neighbors= min(5, len(array)), metric='euclidean')         # pour récupérer 2 acteurs/realisateurs voisins
 

  nn.fit(array)
  personne = array[index_value,:]
  distances, indices = nn.kneighbors(personne)     # on récupère les distances et les indices des acteurs/realisateurs les plus proches

  liste_indices = indices.tolist()         # transformation de l'array des indices en liste
  
  del liste_indices[0][0]                 # supprime le premier de la liste (qui est le film cible)
  liste_indices = liste_indices[0]        # transforme la liste de liste en liste simple
  liste_nconst = []
  liste_nconst = [df.iloc[i]['nconst'] for i in liste_indices] # La liste finale des nconst
  liste_noms = []
  liste_noms = [df_names['Nom'].loc[df_names['nconst']==n].iloc[0] for n in liste_nconst]
  return liste_noms


# Filtrage du dataframe selon le genre sélectionné ----------------------------------------

def suggestion_genre (genre1,genre2,fr) :
    
    
    if genre1 is not None :
       df1 = df_genres.loc[df_genres[genre1] == 1]['tconst']
       df = df1
    if genre2 is not None :
       df2 = df_genres.loc[df_genres[genre2] == 1]['tconst']
       df = df2
    if genre1 is not None and genre2 is not None :

        df = pd.merge(df1,df2,how='inner',on='tconst')

    
    df = pd.merge(df_titres, df, how = 'inner', on = 'tconst')

    if fr :
        df = df.loc[df['original_language']=='fr']

    df = df.sort_values('nbVotes', ascending = False).head(24)
    
    liste_finale = df['tconst'].tolist()
    liste_tmdb = df['id'].tolist() 
    return liste_finale, liste_tmdb


# Affichage des affiches de films ----------------------------------------

def searchMovies(query):
    url = f"https://api.themoviedb.org/3/movie/{query}" #valeur principale de la requête query étant l'index tmdb du film
    params = {  # Paramètres de la requête
        "api_key": 'ca250ec1056f9553bacc5cb920800fec',
        "language": "fr-FR",
        "include_adult": False  
    }
    response = requests.get(url, params=params)
    data = response.json() # Résultat de la requête transformé en json
    return f"https://image.tmdb.org/t/p/original{data['poster_path']}" # On retourne le lien de l'affiche de l'image


def synopsis(query) :
    
    url = f"https://api.themoviedb.org/3/movie/{query}" #valeur principale de la requête query étant l'index tmdb du film
    params = {  # Paramètres de la requête
        "api_key": 'ca250ec1056f9553bacc5cb920800fec',
        "language": "fr-FR",
        "include_adult": False 
    }
    response = requests.get(url, params=params)
    data = response.json() # Résultat de la requête transformé en json
    return data['overview'] # On retourne le synospis du film







######################## AFFICHAGE DE L'APPLICATION ##########################3
df_titres_sorted = df_titres.sort_values('titreVF')  # On classe les titres par ordre alphabétiques
df_names_sorted = df_names.sort_values('Nom') # On classe les noms d'acteurs et réalisateurs par ordre alphabétiques


submitted = 0
requete_trouvee = 0 # Initialisation de la requête, si à 0 rien d'afficher sur l'écran principal
requete2 = 0
with st.sidebar: # Menu sur la gauche pour le choix de la recherche
        
        type_choix = st.selectbox("Quel type de recherche voulez-vous effectuer ?", ['par film','par acteur', 'par réalisateur','par genre']) # Sélection du choix de la recherche
        if type_choix == 'par film':
          liste_choix = df_titres_sorted['titreVF'].tolist()
          titre_test = st.selectbox("Entrez votre titre de film : ", liste_choix, index = None )
          st.session_state.submitted = 0

        elif type_choix == 'par acteur':
          st.session_state.submitted = 0
          liste_choix = df_names_sorted['Nom'].loc[df_names_sorted['commeActeur']!= 'pas_de_film'].tolist()
          titre_test = st.selectbox("Entrez votre acteur : ", liste_choix, index = None )
        elif type_choix == 'par réalisateur':
           st.session_state.submitted = 0
           liste_choix = df_names_sorted['Nom'].loc[df_names_sorted['commeRealisateur']!= 'pas_de_film'].tolist()
           titre_test = st.selectbox("Entrez votre réalisateur : ", liste_choix, index = None )
        elif type_choix == 'par genre' :
          
          with st.form("Recherche") :
            
            liste_choix = liste_genres
            genre1 = st.selectbox("Quel genre de film voulez-vous ?",liste_choix, index = None)
            genre2 = st.selectbox("Voulez-vous sélectionner un deuxième genre ?",liste_choix, index = None)
            fr = st.checkbox("Afficher uniquement les films français")
            if st.form_submit_button("Recherche") :
              st.session_state.submitted = 1
              
            if st.session_state.submitted == 1:
              titre_test = 1
              st.write(f'Vous avez choisi {genre1} et {genre2}')
            else : 
              titre_test = None
            
            

if titre_test is not None :
  if type_choix == 'par film':  # On cherche si c'est un titre de film
    st.session_state.requete_trouvee = 1
    film = id_du_film(titre_test)
    

    # On affiche quelques informations du film choisis :
    
    col1, col2, col3, col4  = st.columns([2,1,2,2])
    with col2 :   
       st.image(searchMovies(df_titres.loc[df_titres['tconst']==film]['id'].iloc[0]), use_container_width=True)  # affiche du film
    with col3 :
       st.header(df_titres.loc[df_titres['tconst']==film]['titreVF'].iloc[0])
       st.write("année de sortie : ",df_titres.loc[df_titres['tconst']==film]['anneeSortie'].iloc[0])   # année sortie
       st.write("Genres : ",df_titres.loc[df_titres['tconst']==film]['genres'].iloc[0])     # liste des genres

       st.write(synopsis(df_titres.loc[df_titres['tconst']==film]['id'].iloc[0]), use_column_width=True)



    st.header('Les films suivants devraient vous plaire :')
    df = df_tri(film)
    films_finaux = df_titres.loc[df_titres['tconst']==film]['tconst_suggestion'].iloc[0]
    imdb = df_titres.loc[df_titres['tconst']==film]['imdb_suggestion'].iloc[0]

  elif type_choix == 'par acteur': # On cherche si c'est genre
    st.session_state.requete_trouvee = 1
    films_finaux, imdb = filmograhie_acteur(titre_test)  
    st.header(f"Filmographie de {titre_test} :")
    liste_noms = meme_genre(df_names.loc[df_names['Nom']==titre_test]['Nom'].iloc[0],'acteur')
    
    requete2 = 1  

  elif type_choix == 'par réalisateur': # On cherche si c'est un acteur ou un réalisateur
    st.session_state.requete_trouvee = 1
    films_finaux, imdb = filmograhie_realisateur(titre_test) 
    st.header(f"Filmographie de {titre_test} :")
    liste_noms = meme_genre(df_names.loc[df_names['Nom']==titre_test]['Nom'].iloc[0],'realisateur')
    
    requete2 = 1 

  elif type_choix == 'par genre' :
    st.session_state.requete_trouvee = 1
    films_finaux, imdb = suggestion_genre(genre1,genre2,fr)

  if st.session_state.requete_trouvee == 1 : # Si on a trouvé un résultat à cette requête on les affiches
    col1, col2, col3, col4, col5, col6  = st.columns(6)
    if len(films_finaux) == 0 :
      st.write("Nous n'avons pas trouvé de film")
    else :
      with col1: 
        if len(films_finaux) >= 0 : # Les if sont pour éviter les messages d'erreur si on n'a pas de films
          st.write(df_titres.loc[df_titres['tconst'] == films_finaux[0]]['titreVF'].iloc[0])
          st.image(searchMovies(imdb[0]), use_container_width=True)
        if len(films_finaux) >= 7 : # Les if sont pour éviter les messages d'erreur si on n'a pas 5 films
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[6]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[6]), use_container_width=True) # use_container_width une fois mis en ligne
        
      with col2: 
        if len(films_finaux) >= 2 :
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[1]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[1]), use_container_width=True)
        if len(films_finaux) >= 8 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[7]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[7]), use_container_width=True)
        
      with col3: 
        if len(films_finaux) >= 3 :
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[2]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[2]), use_container_width=True)
        if len(films_finaux) >= 9 : # Les if sont pour éviter les messages d'erreur si on n'a pas 10 films
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[8]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[8]), use_container_width=True) # use_container_width une fois mis en ligne
        
      with col4: 
        if len(films_finaux) >= 4 :
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[3]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[3]), use_container_width=True)
        if len(films_finaux) >= 10 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[9]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[9]), use_container_width=True)
        
      with col5: 
        if len(films_finaux) >= 5 :
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[4]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[4]), use_container_width=True)
        if len(films_finaux) >= 11 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[10]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[10]), use_container_width=True)
            
      with col6: 
        if len(films_finaux) >= 6 :
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[5]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[5]), use_container_width=True)
        if len(films_finaux) >= 12 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[11]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[11]), use_container_width=True)
            
    if len(films_finaux) > 12 :
      if st.button('Afficher plus') :
        with col1: 
          if len(films_finaux) >= 13 : # Les if sont pour éviter les messages d'erreur si on n'a pas 10 films
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[12]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[12]), use_container_width=True) # use_container_width une fois mis en ligne
          if len(films_finaux) >= 19 : # Les if sont pour éviter les messages d'erreur si on n'a pas 10 films
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[18]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[18]), use_container_width=True) # use_container_width une fois mis en ligne
        with col2: 
          if len(films_finaux) >= 14 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[13]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[13]), use_container_width=True)
          if len(films_finaux) >= 20 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[19]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[19]), use_container_width=True)
        with col3: 
          if len(films_finaux) >= 15 : # Les if sont pour éviter les messages d'erreur si on n'a pas 10 films
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[14]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[14]), use_container_width=True) # use_container_width une fois mis en ligne
          if len(films_finaux) >= 21 : # Les if sont pour éviter les messages d'erreur si on n'a pas 10 films
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[20]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[20]), use_container_width=True) # use_container_width une fois mis en ligne
        with col4: 
          if len(films_finaux) >= 16 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[15]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[15]), use_container_width=True)
          if len(films_finaux) >= 22 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[21]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[21]), use_container_width=True)
        with col5: 
          if len(films_finaux) >= 17 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[16]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[16]), use_container_width=True)
          if len(films_finaux) >= 23 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[22]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[22]), use_container_width=True)
        with col6: 
          if len(films_finaux) >= 18 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[17]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[17]), use_container_width=True)
          if len(films_finaux) >= 24 : 
            st.write(df_titres.loc[df_titres['tconst'] == films_finaux[23]]['titreVF'].iloc[0])
            st.image(searchMovies(imdb[23]), use_container_width=True)
          
    if requete2 == 1 :
      if type_choix == 'par acteur':
        st.header("Autres acteurs qui pourraient vous plaire :")
      else :  
        st.header("Autres réalisateurs qui pourraient vous plaire :")
      
      # premier acteur voisin :
      st.write(f"filmographie de {liste_noms[0]} :")
      if type_choix == 'par acteur':
        filmo, affiche = filmograhie_acteur(liste_noms[0])
      else :  
        filmo, affiche = filmograhie_realisateur(liste_noms[0])
      col1, col2, col3, col4, col5, col6,col7, col8  = st.columns(8)
      if len(filmo) == 0 :
        st.write("Nous n'avons pas trouvé de film")
      else :
        with col1: 
          if len(filmo) >= 0 : # Les if sont pour éviter les messages d'erreur si on n'a pas de films
            st.write(df_titres.loc[df_titres['tconst'] == filmo[0]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[0]), use_container_width=True)
        with col2: 
          if len(filmo) >= 2 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[1]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[1]), use_container_width=True)
        with col3: 
          if len(filmo) >= 3 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[2]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[2]), use_container_width=True)               
        with col4: 
          if len(filmo) >= 4 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[3]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[3]), use_container_width=True)               
        with col5: 
          if len(filmo) >= 5 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[4]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[4]), use_container_width=True)                    
        with col6: 
          if len(filmo) >= 6 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[5]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[5]), use_container_width=True)       
        with col7: 
          if len(filmo) >= 7 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[6]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[6]), use_container_width=True)   
        with col8: 
          if len(filmo) >= 8 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[7]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[7]), use_container_width=True)
      
            # deuxième acteur voisin :
        st.write(f"filmographie de {liste_noms[1]} :")
        if type_choix == 'par acteur':
          filmo, affiche = filmograhie_acteur(liste_noms[1])
        else :  
          filmo, affiche = filmograhie_realisateur(liste_noms[1])
        col1, col2, col3, col4, col5, col6,col7, col8  = st.columns(8)
      if len(filmo) == 0 :
        st.write("Nous n'avons pas trouvé de film")
      else :
        with col1: 
          if len(filmo) >= 0 : # Les if sont pour éviter les messages d'erreur si on n'a pas de films
            st.write(df_titres.loc[df_titres['tconst'] == filmo[0]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[0]), use_container_width=True)
        with col2: 
          if len(filmo) >= 2 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[1]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[1]), use_container_width=True)
        with col3: 
          if len(filmo) >= 3 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[2]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[2]), use_container_width=True)               
        with col4: 
          if len(filmo) >= 4 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[3]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[3]), use_container_width=True)               
        with col5: 
          if len(filmo) >= 5 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[4]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[4]), use_container_width=True)                    
        with col6: 
          if len(filmo) >= 6 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[5]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[5]), use_container_width=True)       
        with col7: 
          if len(filmo) >= 7 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[6]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[6]), use_container_width=True)   
        with col8: 
          if len(filmo) >= 8 :
            st.write(df_titres.loc[df_titres['tconst'] == filmo[7]]['titreVF'].iloc[0])
            st.image(searchMovies(affiche[7]), use_container_width=True)

      if st.button('Autres choix')  :
          # troisième acteur voisin :
        st.write(f"filmographie de {liste_noms[2]} :")
        if type_choix == 'par acteur':
          filmo, affiche = filmograhie_acteur(liste_noms[2])
        else :  
          filmo, affiche = filmograhie_realisateur(liste_noms[2])
        col1, col2, col3, col4, col5, col6,col7, col8  = st.columns(8)
        if len(filmo) == 0 :
          st.write("Nous n'avons pas trouvé de film")
        else :
          with col1: 
            if len(filmo) >= 0 : # Les if sont pour éviter les messages d'erreur si on n'a pas de films
              st.write(df_titres.loc[df_titres['tconst'] == filmo[0]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[0]), use_container_width=True)
          with col2: 
            if len(filmo) >= 2 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[1]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[1]), use_container_width=True)
          with col3: 
            if len(filmo) >= 3 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[2]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[2]), use_container_width=True)               
          with col4: 
            if len(filmo) >= 4 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[3]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[3]), use_container_width=True)               
          with col5: 
            if len(filmo) >= 5 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[4]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[4]), use_container_width=True)                    
          with col6: 
            if len(filmo) >= 6 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[5]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[5]), use_container_width=True)       
          with col7: 
            if len(filmo) >= 7 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[6]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[6]), use_container_width=True)   
          with col8: 
            if len(filmo) >= 8 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[7]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[7]), use_container_width=True)
      
            # quatrième acteur voisin :
          st.write(f"filmographie de {liste_noms[3]} :")
        if type_choix == 'par acteur':
            filmo, affiche = filmograhie_acteur(liste_noms[3])
        else :  
            filmo, affiche = filmograhie_realisateur(liste_noms[3])
        col1, col2, col3, col4, col5, col6,col7, col8  = st.columns(8)
        if len(filmo) == 0 :
          st.write("Nous n'avons pas trouvé de film")
        else :
          with col1: 
            if len(filmo) >= 0 : # Les if sont pour éviter les messages d'erreur si on n'a pas de films
              st.write(df_titres.loc[df_titres['tconst'] == filmo[0]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[0]), use_container_width=True)
          with col2: 
            if len(filmo) >= 2 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[1]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[1]), use_container_width=True)
          with col3: 
            if len(filmo) >= 3 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[2]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[2]), use_container_width=True)               
          with col4: 
            if len(filmo) >= 4 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[3]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[3]), use_container_width=True)               
          with col5: 
            if len(filmo) >= 5 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[4]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[4]), use_container_width=True)                    
          with col6: 
            if len(filmo) >= 6 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[5]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[5]), use_container_width=True)       
          with col7: 
            if len(filmo) >= 7 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[6]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[6]), use_container_width=True)   
          with col8: 
            if len(filmo) >= 8 :
              st.write(df_titres.loc[df_titres['tconst'] == filmo[7]]['titreVF'].iloc[0])
              st.image(searchMovies(affiche[7]), use_container_width=True) 




  else : # On a rien trouvé qui correspond à la requête
    st.write("Nous n'avons pas trouvé de résultat à votre recherche")

  
     

