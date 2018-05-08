
from scipy import spatial
import pandas as pd



def tokenize(doc):
    return doc.lower().split(" ")


def query_similarities(query_string, cosine_sim_threshold, top, bin):
    loaded_tfidf_vectorizer = bin["loaded_tfidf_vectorizer"]
    loaded_tfidf = bin["loaded_tfidf"]
    df_data = bin["df_data"]
    query_tfidf = loaded_tfidf_vectorizer.transform([query_string])
    df_result = pd.DataFrame(columns=['cosine_similarity', 'id', 'sid', 'mid', 'title', 'text'])
    # get the result and add to a new dataframe
    loc_index = 0
    tax_query = query_tfidf.toarray()[0]
    for index_corpus, tax_corpus in enumerate(loaded_tfidf.toarray()):
        cosine_sim = 1 - spatial.distance.cosine(tax_query, tax_corpus)
        if cosine_sim > cosine_sim_threshold:
            df_result.loc[loc_index] = [cosine_sim, df_data.iloc[index_corpus]['id'], df_data.iloc[index_corpus]['sid'],
                                        df_data.iloc[index_corpus]['mid'], df_data.iloc[index_corpus]['title'],
                                        df_data.iloc[index_corpus]['text']]
            loc_index += 1

    # sort the results and pick top results        
    df_result = df_result.sort_values('cosine_similarity', ascending=[False])

    df_result = df_result[:top]

    #     #remove any non-printable characters
    #     df_result['title'] = df_result['title'].apply(lambda x: re.sub(r'[^\x00-\x7f]',r'', x))
    #     df_result['title'] = df_result['title'].apply(lambda x: x.replace(" &#8216", '').replace(" &#8217", ''))
    #     df_result['text'] = df_result['text'].apply(lambda x: re.sub(r'[^\x00-\x7f]',r'', x))
    #     df_result['text'] = df_result['text'].apply(lambda x: x.replace("&#8216;", ' ').replace("&#8217;", ' '))

    return (df_result)


def query_wrapper(query_string, cosine_sim_threshold, top, bin):
    df_result = query_similarities(query_string, cosine_sim_threshold, top, bin)
    return df_result
