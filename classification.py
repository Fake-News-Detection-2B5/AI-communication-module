import re
import ssl
import nltk
import spacy
import string
import pickle
import warnings
import argparse
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from empath import Empath
from nltk import tokenize
import scipy.sparse as sp
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")

class Model:
    def __init__(self, text):
        self.text = text

    def download_dependencies(self):
        pass

    def process_text(self):
        pass

    def predict(self):
        pass

class BiLstm(Model):
    def __init__(self, text):
        super().__init__(text)
        self.model = tf.keras.models.load_model('models/Bi-Lstm/model')
        with open('models/Bi-Lstm/tokenizer.pickle', 'rb') as f:
            self.tokenizer = pickle.load(f)

    def download_dependencies(self):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')

    def process_text(self):
        review = re.sub('[^a-zA-Z]', ' ', self.text)
        review = review.lower()
        review = review.split()
        review = [word for word in review
                  if word not in stopwords.words('english')]
        review = ' '.join(review)
        corpus = [review]
        sequence = self.tokenizer.texts_to_sequences(corpus)
        max_len = max([len(x) for x in sequence])
        self.processed_text = np.array(pad_sequences(sequence, padding='post', maxlen=max_len))

    def predict(self):
        if self.processed_text.any():
            prediction = self.model.predict_classes(self.processed_text)
            if prediction[0] == 0:
                return "false"
            elif prediction[0] == 1:
                return "true"
            elif prediction[0] == 2:
                return "partially false"
            elif prediction[0] == 3:
                return "other"


class Sentiment(Model):
    def __init__(self, text):
        super().__init__(text)
        self.model = tf.keras.models.load_model('models/SentimentAnalysis/model')
        with open('models/SentimentAnalysis/tokenizer.pickle', 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    def download_dependencies(self):
        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords')

    def process_text(self):
        self.text = re.sub(self.TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
        tokens = []
        stop_words = stopwords.words('english')
        stemmer = SnowballStemmer('english')
        for token in text.split():
            if token not in stop_words:
                tokens.append(stemmer.stem(token))
        self.text = " ".join(tokens)
        self.data = pad_sequences(self.tokenizer.texts_to_sequences(self.text), maxlen=700)

    def predict(self):
        if self.data.any():
            prediction = self.model.predict_classes(self.data)
            if prediction[0] == 0:
                return "partially false"
            elif prediction[0] == 1:
                return "false"
            elif prediction[0] == 2:
                return "true"
            elif prediction[0] == 3:
                return "other"

class Three_Layer(Model):
	def __init__(self, text):
		super().__init__(text)

	def download_dependencies(self):
		try:
			nltk.data.find('stopwords')
			nltk.data.find('wordnet')
			nltk.data.find('words')
		except LookupError:
			nltk.download('stopwords')
			nltk.download('wordnet')
			nltk.download('words')

	def process_text(self):
		self.clean_text = clean_text(self.text)
       
		nlp = spacy.load('en_core_web_sm')
		pos_tags_column = []

		for text in self.text.split():
			pos_tags = []
			doc = nlp(text)
			for token in doc:
				pos_tags.append(token.pos_)
			all_pos_tags = ' '.join(pos_tags)
			pos_tags_column.append(all_pos_tags)
    
		self.POS_text = pos_tags_column

		lexicon = Empath()
		semantic = []
		count = 0

		d = lexicon.analyze(self.text.split(), normalize=False)
		x = []
		for key, value in d.items():
			x.append(value)
		x = np.asarray(x)
		semantic.append(x)

		self.semantic_text = semantic

		categories = []
		a = lexicon.analyze("")
		for key, value in a.items():
			categories.append(key)

		sem = []
		a = []
		for j in range(len(semantic[0])):
			for k in range(int(semantic[0][j])):
				a.append(categories[j])
		b = " ".join(a)
		sem.append(b)

		self.semantics_text = sem

		X_test_text = self.clean_text
		X_test_POS = self.POS_text
		X_test_sem = self.semantics_text

		empty = ""
		for pos in X_test_POS:
			empty += pos + " "
		X_test_POS = [empty]


		loaded_tfidf = pickle.load(open('./models/Three_Layer/vectorizers/tfidf_pickle.sav', 'rb'))
		loaded_pos = pickle.load(open('./models/Three_Layer/vectorizers/pos_pickle.sav', 'rb'))
		loaded_sem = pickle.load(open('./models/Three_Layer/vectorizers/sem_pickle.sav', 'rb'))
       
		tfidf_test = loaded_tfidf.transform([X_test_text])
		pos_tfidf_test = loaded_pos.transform(X_test_POS)
		sem_tfidf_test = loaded_sem.transform(X_test_sem)

		text_w = 0.5 * 3
		pos_w = 0.15 * 3
		sem_w = 0.35 * 3

		tfidf_test *= text_w
		pos_tfidf_test *= pos_w
		sem_tfidf_test *= sem_w

		diff_n_rows = pos_tfidf_test.shape[0] - tfidf_test.shape[0]
		d = sp.vstack((tfidf_test, sp.csr_matrix((diff_n_rows, tfidf_test.shape[1]))))
		e = sp.hstack((pos_tfidf_test, d))

		diff_n_rows = e.shape[0] - sem_tfidf_test.shape[0]
		d = sp.vstack((sem_tfidf_test, sp.csr_matrix((diff_n_rows, sem_tfidf_test.shape[1]))))

		self.X_test = sp.hstack((e, d))

	def load_model(self):
		self.loaded_model = pickle.load(open('./models/Three_Layer/three_layer_pickle.sav', 'rb'))

	def predict(self):
		self.load_model()
		return self.loaded_model.predict(self.X_test)[0].lower()

def parse_args():
    parser = argparse.ArgumentParser(description='Fake News Classification')
    parser.add_argument('text', metavar='text', type=str, nargs='+', help='Text to be classified')
    args = parser.parse_args()
    return ' '.join(args.text)

def clean_text(text):
	text = re.sub('['+string.punctuation+']','', text)
	text = re.sub(r"[-()\"#/@’;:<>{}`+=~|.!?,]", '', text)
	text = text.lower().split()

	stops = set(stopwords.words("english"))
	text = [w for w in text if w not in stops]
	text = " ".join(text)
  
	text = re.sub(r'[^a-zA-Z\s]', u'', text, flags=re.UNICODE)
  
	text = text.split()
	l = WordNetLemmatizer()
	lemmatized_words = [l.lemmatize(word) for word in text if len(word) > 2]
	text = " ".join(lemmatized_words)
    
	return text


if __name__ == '__main__':
    text = parse_args()

    # Bi LSTM
    bilstm = BiLstm(text)
    bilstm.download_dependencies()
    bilstm.process_text()
    prediction_bilstm = bilstm.predict()

    # Sentiment Analysis
    sentiment = Sentiment(text)
    sentiment.download_dependencies()
    sentiment.process_text()
    prediction_sentiment = sentiment.predict()

    # Three-Layer
    three_layer = Three_Layer(text)
    three_layer.download_dependencies()
    three_layer.process_text()
    prediction_three_layer = three_layer.predict()

    c = Counter([prediction_bilstm, prediction_sentiment, prediction_three_layer])
    value, count = c.most_common()[0]
    if value:
        with open ('scor.txt', 'w') as f:
            f.write(value)
