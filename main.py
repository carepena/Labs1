from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

#mostramos un mensaje de bienvenida
def index():
    return {'Proyecto Labs1 de CARLOS EDUARDO PEÑA'}


#abrimos el nuevo csv para realizar las funciones y hacer las pruebas
df = pd.read_csv('df_merged_final.csv', low_memory=False)


#esta función devuelve el año de lanzamiento con más horas jugadas para un género determinado.
@app.get('/PlayTimeGenre/')
def PlayTimeGenre(genre: str) -> dict:
    #convierte el género a mayúscula inicial para que sea consistente con el nombre de las columnas del dataframe.
    genre = genre.capitalize()
    #filtra el dataframe para que solo contenga los juegos del género especificado
    #luego, calcula el total de horas jugadas por año y devuelve el año con el valor más alto.
    genre_df = df[df[genre] == 1]
    year_playtime_df = genre_df.groupby('posted year')['playtime_forever'].sum().reset_index()
    max_playtime_year = year_playtime_df.loc[year_playtime_df['playtime_forever'].idxmax(), 'posted year']
    #devuelve un diccionario con el género y el año de lanzamiento con más horas jugadas.
    return{'Genero': genre, 'Año de lanzamiento con mas horas jugadas para Genero:': int(max_playtime_year)}

#esta función devuelve el usuario con más horas jugadas para un género determinado.
@app.get('/UserForGenre/')
def UserForGenre(genre: str) -> dict:
    #convierte el género a mayúscula inicial para que sea consistente con el nombre de las columnas del dataframe.
    genre = genre.capitalize()
    #filtra el dataframe para que solo contenga los juegos del género especificado
    #luego, calcula el total de horas jugadas por usuario y devuelve el usuario con el valor más alto.
    genre_df = df[df[genre] == 1]
    max_playtime_user = genre_df.loc[genre_df['playtime_forever'].idxmax(), 'user_id']
    year_playtime_df = genre_df.groupby('posted year')['playtime_forever'].sum().reset_index()
    playtime_list = year_playtime_df.to_dict(orient='index')
    #devuelve un diccionario con el género, el usuario con más horas jugadas y una lista con las horas jugadas por año.
    result = {
        'Usuario con mas horas jugadas para Genero' + genre: max_playtime_user,
        'Horas jugadas': playtime_list}
    return result


#funciones `UsersRecommend()` y `UsersNotRecommend()
#estas funciones devuelven los tres juegos con la calificación más alta (2) para un año determinado
#la función `UsersRecommend()` solo considera los juegos que fueron recomendados por el usuario, 
#mientras que la función `UsersNotRecommend()` solo considera los juegos que no fueron recomendados por el usuario.
@app.get('/UsersRecommend/')
def UsersRecommend(year: int) -> dict:
    df_filtrado = df[(df['posted year']== year) & (df['recommend'] == True) & (df['sentiment_score'] == 2)]
    if df_filtrado.empty:
        return {'error': 'no encontrado'}
    df_ordenado = df_filtrado.sort_values(by='sentiment_score', ascending=False)
    top_3_resenas = df_ordenado.head(3)
    resultado = {
        'Puesto 1': top_3_resenas.iloc[0]['title'],
        'Puesto 2': top_3_resenas.iloc[1]['title'],
        'Puesto 3': top_3_resenas.iloc[2]['title'],
    }
    return resultado

@app.get('/UsersNotRecommend/')
def UsersNotRecommend(year: int)-> dict:
    df_filtrado = df[(df['posted year']== year) & (df['recommend'] == False) & (df['sentiment_score'] <=1)]
    if df_filtrado.empty:
        return {'error': 'no encontrado'}
    df_ordenado = df_filtrado.sort_values(by='sentiment_score', ascending=False)
    top_3_resenas = df_ordenado.head(3)
    resultado = {
        'Puesto 1': top_3_resenas.iloc[0]['title'],
        'Puesto 2': top_3_resenas.iloc[1]['title'],
        'Puesto 3': top_3_resenas.iloc[2]['title'],
    }
    return resultado

#esta función devuelve un diccionario con el número de juegos con cada calificación de sentimiento para un año determinado.
@app.get('/sentiment_score/')
def sentiment_analysis(year: int) -> dict:
    filtra_df = df[df['posted year']== year]
    sentiment = filtra_df['sentiment_score'].value_counts()
    result = {
        'Positivo': int(sentiment.get(2, 0)),
        'Neutral': int(sentiment.get(1, 0)),
        'Negativo': int(sentiment.get(0, 0))
    }
    return result


muestra = df.head(25000)
tfidf = TfidfVectorizer(stop_words='english')
muestra = muestra.fillna('')

tfidf_matri = tfidf.fit_transform(muestra['review'])
cosine_similarity = linear_kernel(tfidf_matri, tfidf_matri)


#esta función devuelve una lista con los tres juegos más similares a los juegos que el usuario ha jugado.
@app.get('/recomendacion_usuario/{id_juego}')
def recomendacion_juego(id_juego: int) -> dict:
    if id_juego not in muestra['id'].values:
        return {'mensaje': 'No existe el id del juego.'}
    titulo = muestra.loc[muestra['id']== id_juego, 'title'].iloc[0]
    idx = muestra[muestra['title'] == titulo].index[0]
    sim_cosine = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True)
    sim_ind = [i for i, _ in sim_scores[1:6]]
    sim_juegos = muestra['title'].iloc[sim_ind].values.tolist()

    return {'juegos recomendados': list(sim_juegos)}