import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def top_10_tfidf(csv_file_path, n_words):
    '''Function that takes in a csv file and gives the top n results
        of the TF-IDF on each book of the Bible

        Note: 
            Format of CSV must be in format:
                Verse ID, Book Name, Book Number, Chapter, Verse, Text

        Args:
            csv_file_path (str): The file path of the csv,
            n_words (int): The top n results of each book
        
        Returns:
            DataFrame: A dataframe with the results for each book of the Bible
    
    '''
    bible_csv = pd.read_csv(csv_file_path)
    patt = r"[^\w,;:'()\s.!?]"
    cleaned_csv = bible_csv['Text'].str.replace(patt, '', regex=True)
    bible_csv = bible_csv.assign(clean=cleaned_csv)
    bible_csv = (
            bible_csv
                 .groupby(
                     ['Book Number', 'Book Name']
                     )[['Book Number', 'Book Name', 'clean']]
                 .apply(lambda x: x['clean'].str.cat(sep=' '))
                 .to_frame()
                 .rename(columns={0:'Text'})
                 )

    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    X = tfidf.fit_transform(bible_csv['Text'])
    tfidf_df = pd.DataFrame(
        data=X.toarray(), 
        columns=tfidf.get_feature_names_out()
        )
    
    top_words = (
            tfidf_df
                 .apply(lambda x: list(x.sort_values(ascending=False)[:n_words].index), axis=1)
                 .to_frame()
                 .rename(columns={0:f'Top Words ({csv_file_path[5:-4].upper()})'})
                 )
    return top_words

if __name__ == '__main__':
    test_csv = 'data/asv.csv'
    df = top_10_tfidf(test_csv, 12)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(top_10_tfidf(test_csv))
    print(df.to_string())