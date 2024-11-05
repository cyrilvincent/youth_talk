from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
# pd.set_option("max_rows", 600)
from pathlib import Path
import glob

# D'après https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/03-TF-IDF-Scikit-Learn.html

# directory_path = "data/US_Inaugural_Addresses/"
# text_files = glob.glob(f"{directory_path}/*.txt")
# text_titles = [Path(text).stem for text in text_files]
#
# tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english') #content
# tfidf_vector = tfidf_vectorizer.fit_transform(text_files)
# tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names_out())
# tfidf_df.loc['00_Document Frequency'] = (tfidf_df > 0).sum()
# tfidf_slice = tfidf_df[['government', 'borders', 'people', 'obama', 'war', 'honor', 'foreign', 'men', 'women', 'children']]
# print(tfidf_slice.sort_index().round(decimals=2))
#
# tfidf_df = tfidf_df.drop('00_Document Frequency', errors='ignore')
# tfidf_df.stack().reset_index()
# tfidf_df = tfidf_df.stack().reset_index()
# tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'document','level_1': 'term', 'level_2': 'term'})
# print(tfidf_df.sort_values(by=['tfidf'], ascending=[False]).head(20))
#
#
# top_tfidf = tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(10)
# print(top_tfidf[top_tfidf['term'].str.contains('women')])
#
# print(top_tfidf[top_tfidf['document'].str.contains('obama')])

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
import numpy as np
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
columns=vectorizer.get_feature_names_out()
df = pd.DataFrame(X.toarray(), columns=columns)
nb = len(df)
df2=df.stack()
print(df)

# df=df.rename(index={0: "x", 1: "y", 2: "z"})
# for col, row in zip(list(columns) * nb, df2):
#     print(col, row)

for index, row in df.iterrows():
    for col in columns:
        print(index, col, row[col])






