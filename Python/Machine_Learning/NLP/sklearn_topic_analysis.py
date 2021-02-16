from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

"""
Jonathan Rees, Data Science Campus, Office for National Statistics
November 2019
Please contact datasciencecampus@ons.gov.uk for more information
"""

#############################
#      Pipelines            #
#############################

# Pipeline to take job data DF and perform topic analysis.
# Can assign new dataframe with added topic column, and output NMF or lda model
# new_df, model = assign_nmf_topics(salaries, 3)
#
# Can add column with topic number with the following example;
# assign_nmf_topics(salaries, 3)


def assign_nmf_topics(df, n_topics, colname='Description', ngram_range=(1, 3)):
    """
    pipeline to assign topics to a dataframe with a free text column
    free text column should be string rather than list of tokens
    args :
        df : dataframe to be analysed for topics
        num_topics : number of topics to be assigned
        colname : name of free text column
        ngram_range : range of ngrams.
        e.g. (2, 3) includes only bigrams and trigrams. defaults to (1, 3)
    returns:
        df : new dataframe with aditional topic column
        model : nmf model object
    """
    # set up tfidf vectorizer and document matrix
    tfidf_vect, doc_term_matrix = tfidf_setup(df, colname,
                                              ngram_range=ngram_range)

    # set up nmf model
    nmf = build_nmf_model(n_topics, doc_term_matrix)

    # print out top words from each topic
    print_topics(nmf, tfidf_vect)

    # fit each entry to most related topic
    fit_topics(df, nmf, doc_term_matrix)
    return df, nmf


def assign_lda_topics(df, n_topics, colname='Description', ngram_range=(1, 3)):
    """
    pipeline to assign topics to a dataframe with a free text column
    free text column should be string rather than list of tokens
    args :
        df : dataframe to be analysed for topics
        num_topics : number of topics to be assigned
        colname : name of free text column
        ngram_range : range of ngrams.
        e.g. (2, 3) includes only bigrams and trigrams. defaults to (1, 3)
    returns:
        df : new dataframe with aditional topic column
        model : lda model object
    """
    # set up count vectorizer and document term matrix
    count_vect, doc_term_matrix = cv_setup(df, colname,
                                           ngram_range=ngram_range)

    # set up lda moel
    lda = build_lda_model(n_topics, doc_term_matrix)

    # print out top words from each topic
    print_topics(lda, count_vect)

    # fit each entry to most related topic
    fit_topics(df, lda, doc_term_matrix)
    return df, lda

#############################
#         Functions         #
#############################


def tfidf_setup(df, colname='Description', maxdf=0.8, mindf=2,
                stops='english', ngram_range=(1, 3)):
    """
    initialises and fits the tfidf vectorizer for nmf.
    args :
        df : pandas dataframe with free text column
        colname : string name of column to analyse. defaults to 'Description'
        maxdf : max number of documents for a word to appear in.
                defaults to 80%
        mindf : minimum number of documents for a word to appear in.
                defaults to 2 documents
        stops : list of stopwords to be used by TF:IDF. defaults to english
        ngram_range : range of ngrams.
        e.g. (2, 3) includes only bigrams and trigrams. defaults to (1, 3)
    returns : vectorizer and document term matrix
    """
    # set up TFIFD vectorizer
    tfidf_vect = TfidfVectorizer(max_df=maxdf,
                                 min_df=mindf,
                                 stop_words=stops, 
                                 ngram_range=ngram_range)

    doc_term_matrix = tfidf_vect.fit_transform(
            df[colname].values.astype('U'))
    return tfidf_vect, doc_term_matrix


def cv_setup(df, colname='Description', maxdf=0.8, mindf=2,
             stops='english', ngram_range=(1, 3)):
    """
    initialises and fits the count vectorizer for lda.
    args :
        df : pandas dataframe with free text column
        colname : string name of column to analyse. defaults to 'Description'
        maxdf : max number of documents for a word to appear in.
                defaults to 80%
        mindf : minimum number of documents for a word to appear in.
                defaults to 2 documents
        stops : list of stopwords to be used by cv. defaults to english
        ngram_range : range of ngrams.
        e.g. (2, 3) includes only bigrams and trigrams. defaults to (1, 3)
    returns : vectorizer and document term matrix
    """
    count_vect = CountVectorizer(max_df=maxdf,
                                 min_df=mindf,
                                 stop_words=stops,
                                 ngram_range=ngram_range)

    doc_term_matrix = count_vect.fit_transform(
            df[colname].values.astype('U'))
    return count_vect, doc_term_matrix


def build_nmf_model(n_topics, dtm):
    """
    creates nmf model
    args:
        n_topics : number of topics to be assigned
    returns:
        nmf topic analysis model
    """
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(dtm)
    return nmf


def build_lda_model(n_topics, dtm):
    """
    creates lda model
    args:
        n_topics : number of topics to be assigned
    returns:
        lda topic analysis model
    """
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    return lda


def print_topics(model, vect, numwords=10):
    """
    prints out top n words from each topic from the topic analysis model
    args :
        model : topic analysis model to be used
        numwords : integer for how many words to be displayed. Starts at
                   highest value. defaults to 10
        vect : vectorizer item. either TFIDF or c_v
    """
    for i, topic in enumerate(model.components_):
        print(f'Top 10 words for topic #{i}:')
        print([vect.get_feature_names()[i] for i in
               topic.argsort()[-numwords:]])
        print('\n')


def fit_topics(df, model, dtm):
    """
    fits each entry in dataframe to the most relevant topic
    args :
        df : pandas dataframe to be analysed
        model : model to be used
        dtm : document term matrix
    returns : dataframe with column 'Topic' including the number of the topic
              which best fits the text.
    """
    topic_values = model.transform(dtm)
    df['Topic'] = topic_values.argmax(axis=1)
    return df
