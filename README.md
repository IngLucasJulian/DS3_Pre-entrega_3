# Pre-entrega 3 - Clasificador de noticias con TF-IDF

Lucas Nahuel Julian - Data Science III

En esta entrega armé un clasificador de noticias usando TF-IDF y un modelo de Machine Learning.
Uso el mismo dataset y el mismo preprocesamiento que en la pre-entrega 2.

## Dataset

AG News
- data/ag_news_train.csv (8000 noticias)
- data/ag_news_test.csv (2000 noticias)

Las clases están balanceadas y usé los splits de train y test como vienen.

## Cómo ejecutarlo

bash
pip install -r requirements.txt


Después abrir Clasificador_TFIDF_JulianLucas.ipynb y ejecutar todas las celdas.

Versiones que usé: Python 3.13, scikit-learn 1.9.1, pandas 3.0.5 y spaCy 3.8.18 (el resto está en
requirements.txt).

## Qué hace el notebook

1. Carga los datos de train y test.
2. Limpia el texto con las funciones de la pre-entrega 2 (clean_text y preprocess_text): saco HTML,
   links, números y stop-words, y lematizo con spaCy.
3. Vectoriza con TfidfVectorizer. Hago fit_transform solo con train y transform con test, para no
   tener data leakage.
4. Entreno el modelo y lo evalúo con classification_report y la matriz de confusión.

## Parámetros del vectorizador

- ngram_range=(1, 2): uso palabras sueltas y bigramas, así toma cosas como "oil price" o "prime minister".
- max_features=10000: con bigramas el vocabulario llegaba a casi 147000 términos, así que lo limité.
  Probé con 5000, 10000, 20000 y sin límite y los resultados fueron casi iguales (F1 entre 0,882 y
  0,885), por eso me quedé con 10000, que es más liviano.
- sublinear_tf=True: para que una palabra repetida muchas veces en la misma noticia no pese de más.

Las pruebas las hice con validación cruzada dentro de train, así el test lo uso recién al final.

## Por qué elegí Naive Bayes

Comparé los tres modelos que propone la consigna con validación cruzada:

| Modelo | F1 macro |
|---|---|
| Naive Bayes | 0,889 |
| Regresión Logística | 0,883 |
| SVM lineal | 0,882 |

Elegí Naive Bayes porque dio el mejor resultado, es rápido de entrenar y es un modelo muy usado como base
para clasificar texto. Además da probabilidades para cada clase.

## Resultados

En test el modelo obtuvo una accuracy de 0,896 (en train 0,927, así que no hay mucho overfitting).

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| Business | 0,84 | 0,86 | 0,85 |
| Sci_Tech | 0,87 | 0,88 | 0,88 |
| Sports | 0,95 | 0,97 | 0,96 |
| World | 0,92 | 0,88 | 0,90 |

![Matriz de confusión](matriz_confusion.png)

## Qué categorías le cuestan más

- Sports es la más fácil, casi no se equivoca porque usa palabras muy propias (game, team, season).
- Business es la más difícil. Se confunde mucho con Sci_Tech y al revés, porque las noticias de
  empresas de tecnología hablan de ventas y ganancias, y muchas de negocios hablan de productos
  tecnológicos.
- World a veces se confunde con Business cuando la noticia es sobre la economía de un país.

