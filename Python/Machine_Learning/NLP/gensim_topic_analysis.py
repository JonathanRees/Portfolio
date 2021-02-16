from modules import textpreprocess as tpp
from modules import topicanalysis as ta
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from numpy import array
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.nmf import Nmf
salaries = pd.read_csv(FILEPATH, encoding = 'unicode escape')

"""
Jonathan Rees, Data Science Campus, Office for National Statistics
November 2019
Please contact datasciencecampus@ons.gov.uk for more information

This pipeline is less developed than the SKLEARN version, as I changed projects part way through its development. I mean to flesh it out more in the future.
"""

#################
#  Pipeline     #
#################


def lda_coherence_pipeline(df, num_topics=7):
    """
    pipeline to calculate lda coherence for a range of topic numbers,
    up to a maximum
    args:
        df : dataframe with free text column
        num_topics : maximum number of topics to calculate coherence scores for
        defaults to 7
    outputs :
        graph showing coherence scores for each number of topics
    """
    docs = prepare_docs_from_df(df)
    dictionary = create_dictionary(docs)
    corpus = create_corpus(dictionary, docs)
    model_list, coherence_values = compute_lda_coherence_values(dictionary\
                                                                =dictionary,
                                                            corpus=corpus,
                                                            texts=docs,
                                                            start=2,
                                                            limit=num_topics,
                                                            step=1)
    plot_coherence_values(coherence_values, num_topics = num_topics)


def nmf_coherence_pipeline(df, num_topics=7):
    """
    pipeline to calculate nmf coherence for a range of topic numbers,
    up to a maximum
    args:
        df : dataframe with free text column
        num_topics : maximum number of topics to calculate coherence scores for
        defaults  to 7
    outputs :
        graph showing coherence scores for each number of topics
    """
    docs = prepare_docs_from_df(df)
    dictionary = create_dictionary(docs)
    corpus = create_corpus(dictionary, docs)
    model_list, coherence_values = compute_nmf_coherence_values(dictionary\
                                                                =dictionary,
                                                            corpus=corpus,
                                                            texts=docs,
                                                            start=2,
                                                            limit=num_topics,
                                                            step=1)
    plot_coherence_values(coherence_values, num_topics=num_topics)


docs = prepare_docs_from_df(salaries)
dictionary = create_dictionary(docs)
corpus = create_corpus(dictionary, docs)
lda_model = build_lda_model(corpus, dictionary, 3)
ta.print_topics()


#################
#  Functions    #
#################

# Set parameters.
num_topics = 3
chunksize = 500
passes = 20
iterations = 400
eval_every = 1


def prepare_docs_from_df(df, colname = 'Description', newcolname = 'processed_text'):
    """
    """
    df[newcolname] = df[colname].apply(lambda x: tpp.pre_process(x, tokenize=True))
    docs = array(df['processed_text'])
    return docs

salaries['processed'] = salaries['Description'].apply(lambda x: tpp.pre_process(x, tokenize=True))
docs = array(salaries['processed'])
return docs

def create_dictionary(docs):
            
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=10, no_above=0.2)
    return dictionary


def create_corpus(dictionary, docs):
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    print(corpus[:1])
    return corpus


def build_lda_model(corpus, dictionary, num_top = 5):

    temp = dictionary[0]  # only to "load" the dictionary.
    id2word = dictionary.id2token

    lda_model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                           alpha='auto', eta='auto', \
                           iterations=iterations, num_topics=num_top, \
                           passes=passes, eval_every=eval_every)
    return lda_model


def build_nmf_model(corpus):
    nmf_model = Nmf(corpus, num_topics=10)
    return nmf_model


def print_coherence_values(lda_model, dictionary):
    # Compute Coherence Score using c_v
    coherence_model_lda = CoherenceModel(model = lda_model,
                                         texts = docs,
                                         dictionary = dictionary,
                                         coherence = 'c_v')
    coherence_lda_cv = coherence_model_lda.get_coherence()
    print('\nCoherence Score with C_V : ', coherence_lda_cv)

    # Compute Coherence Score using UMass
    coherence_model_lda = CoherenceModel(model = lda_model,
                                         texts = docs,
                                         dictionary = dictionary,
                                         coherence = "u_mass")
    coherence_lda_umass = coherence_model_lda.get_coherence()
    print('\nCoherence Score with UMass : ', coherence_lda_umass)


def compute_lda_coherence_values(dictionary, corpus, texts,
                             limit, start=2, step=1):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word=dictionary,
                       num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts,
                                        dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def compute_nmf_coherence_values(dictionary, corpus, texts,
                             limit, start=2, step=1):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_top in range(start, limit, step):
        model = Nmf(corpus=corpus, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts,
                                        dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def plot_coherence_values(coherence_values, num_topics = 7):
    limit= num_topics; start=2; step=1;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num),
                                                                  round(prop_topic,4), topic_keywords]),
    ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic',
                              'Perc_Contribution',
                              'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model,
                                                  corpus=corpus, texts=docs)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No',
                             'Dominant_Topic',
                             'Topic_Perc_Contrib',
                             'Keywords', 'Text']

# Show
df_dominant_topic.head(10)
