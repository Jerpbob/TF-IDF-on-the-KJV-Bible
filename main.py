import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def top_10_tfidf(csv_file_path):
    
    bible_csv = pd.read_csv(csv_file_path)
    patt = r"[^\w,;:'()\s.!?]"
    cleaned_csv = bible_csv['Text'].str.replace(patt, '', regex=True)
    bible_csv = bible_csv.assign(clean=cleaned_csv)
    bible_csv = (bible_csv
                 .groupby(
                     ['Book Number', 'Book Name']
                     )[['Book Number', 'Book Name', 'clean']]
                 .apply(lambda x: x['clean'].str.cat(sep=' '))
                 .to_frame()
                 .rename(columns={0:'Text'}))

    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    X = tfidf.fit_transform(bible_csv['Text'])
    tfidf_df = pd.DataFrame(
        data=X.toarray(), 
        columns=tfidf.get_feature_names_out()
        )
    
    top_words = (tfidf_df
                 .apply(lambda x: list(x.sort_values(ascending=False)[:10].index), axis=1)
                 .to_frame()
                 .rename(columns={0:f'Top Words ({csv_file_path[5:-4].upper()})'}))
    return top_words