###############################################################################
# FILENAME: TextAnalytics.py
# AUTHOR: Matthew Hartigan
# DATE: 14-June-2021
# DESCRIPTION: A python script to analyze the words in input text files.
###############################################################################


# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplt as plt
import os
import nltk
from wordcloud import WordCloud
from nltk import pos_tag
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from corpus_toolkit import corpus_tools as ct


# README: if receiving error messages from nltk, may need to launch in python console and download libraries


class TextAnalytics():

    # CONSTRUCTORS
    def __init__(self, file_directory):
        self.input_dir = file_directory
        self.df = None
        self.dtm = None
        self.dtm_df = None

    def import_raw_data(self):
        file_names = os.listdir(self.input_dir)
        file_name_and_text = {}
        for file in file_names:
            with open(self.input_dir + file, 'r', encoding='utf-8') as target_file:
                file_name_and_text[file] = target_file.read()
        self.df = pd.DataFrame.from_dict(file_name_and_text, orient='index').reset_index()
        self.df = self.df.rename(index=str, columns={'index': 'file_name', 0: 'text'})

    def add_columns(self):
        year = []
        quarter = []
        for iteration, filename in enumerate(self.df['file_name']):
            search_term = 'Q'
            str_index = filename.find(search_term)
            quarter.append(filename[str_index-1])
            year.append('20' + filename[(str_index+1):(str_index+3)])
        self.df['year'] = year
        self.df['quarter'] = quarter

    def corpus(self):
        word_count = []
        for iteration, filename in enumerate(self.df['file_name']):
            word_count.append(len(self.df.iloc[iteration]['text'].split()))
        self.df['word_count'] = word_count
        print('Max word count is: ' + str(self.df['word_count'].max()))
        print('It occurs in ' + self.df.sort_values(by=['word_count'], ascending=False.reset_index().iloc[0]['file_name']))
        print()

    def process_text_data(self):
        processed_text = {}
        lemmatizer = WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english')
        newStopWords = []
        stopwords.extend(newStopWords)

        for iteration, entry in enumerate(self.df['text']):
            # Tokenize (removes punctuation adn shifts to all lower case)
            tokenizer = RegexpTokenizer(r'\w+')
            processed_text[iteration] = tokenizer.tokenize(entry.lower())
            # Remove numbers
            processed_text[iteration] = [item for item in processed_text[iteration] if not item.isdigit()]
            # Remove stopwords
            processed_text[iteration] = [word for word in processed_text[iteration] if word not in stopwords]
            # Lemmatize
            processed_text[iteration] = " ".join([lemmatizer.lemmatize(i) for i in processed_text[iteration]])
        self.df['lemmatized_text'] = list(processed_text.values())    # append to dataframe

        # Create DTM
        cv = CountVectorizer(ngram_range=(1, 1))
        self.dtm = cv.fit_transform(self.df['lemmatized_text'])
        words = np.array(cv.get_feature_names())
        print(pd.DataFrame.from_records(self.dtm[:5, :5].A, columns=words[:5]))
        self.dtm_df = pd.DataFrame.from_records(self.dtm.A, columns=words)

        # Analyze frequency
        freqs = self.dtm.sum(axis=0).A.flatten()
        index = np.argsort(freqs)[-50:]
        print(list(zip(words[index], freqs[index])))
        WordFreq = pd.DataFrame.from_records((list(zip(words[index], freqs[index]))))
        WordFreq.columns = ['Word', 'Freq']

        # Plot bar graph
        fig, ax = plt.subplots(figsize=(8, 8))
        WordFreq.sort_values(by='Freq').plot.barh(x='Word', y='Freq', ax=ax, color='gray')

        # Look at metadata (word count over time)
        self.df['year'] = self.df['year'].astype(str).astype(int)
        self.df.plot.line(x='year', y='word_count')

        # Generate word cloud
        data = dict(zip(WordFreq['Word'].tolist(), WordFreq['Freq'].tolist()))
        wordcloud = WordCloud().generate_from_frequencies(data)

        # Plotting
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

        # LDA
        def print_topics(model, count_vectorizer, n_top_words):
            words = count_vectorizer.get_feature_names()
            for topic_idx, topic in enumerate(model.components_):
                print('\nTopic #%d: ' % topic_idx)
                print(' '.join([words[i] for i in topic.argsort()[:n_top_words - 1: -1]]))

        # Set parameters
        number_topics = 5
        number_words = 5

        # Create and fit LDA model
        lda = LDA(n_components=number_topics, n_jobs=-1)
        lda.fit(self.dtm)
        print('Topics found via the LDA:')
        print_topics(lda, cv, number_words)

    def compare(self):
        groupA = self.df.copy()
        groupA['quarter'] = groupA[groupA['quarter'].astype(int) < 2]['quarter']
        groupA = groupA.dropna()
        groupB = self.df.copy()
        groupB['quarter'] = groupB[groupB['quarter'].astype(int) < 2]['quarter']
        groupB = groupB.dropna()

        # Create DTM
        for count, group in enumerate([groupA, groupB]):
            cv2 = CountVectorizer(ngram_range=(1, 1))
            dtm = cv2.fit_transform(group['lemmatized_text'])
            words = np.array(cv2.get_feature_names())
            dtm_df = pd.DataFrame.from_records(dtm.A, columns=words)

            # Analyze frequency
            freqs = dtm.sum(axis=0).A.flatten()
            index = np.argsort(freqs)[-50:]
            WordFreq = pd.DataFrame.from_records((list(zip(words[index], freqs[index]))))
            WordFreq.columns = ['Word', 'Freq']

            # Plot bar graph
            fig, ax = plt.subplots(figsize=(8, 8))
            WordFreq.sort_values(by='Freq').plot.barh(x='Word', y='Freq', ax=ax, color='gray')
            plt.title('Group ' + str(count))
            plt.figure()
            plt.show()


if __name__ == '__main__':
    input_dir = 'FIXME'
    assignment = TextAnalytics(input_dir)
    assignment.import_raw_data()
    assignment.add_columns()
    assignment.corpus()
    assignment.process_text_data()
    assignment.compare()
