import re
import ssl
import nltk
# import spacy
import string
import pickle
import warnings
# import argparse
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
# from empath import Empath
from nltk import tokenize
# import scipy.sparse as sp
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
# from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import torch
# from transformers import RobertaModel, RobertaTokenizer, BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, \
#     get_linear_schedule_with_warmup
# from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

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


# class Sentiment(Model):
#     def __init__(self, text):
#         super().__init__(text)
#         self.model = tf.keras.models.load_model('models/SentimentAnalysis/model')
#         with open('models/SentimentAnalysis/tokenizer.pickle', 'rb') as f:
#             self.tokenizer = pickle.load(f)
#         self.TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

#     def download_dependencies(self):
#         try:
#             nltk.data.find('stopwords')
#         except LookupError:
#             nltk.download('stopwords')

#     def process_text(self):
#         self.text = re.sub(self.TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
#         tokens = []
#         stop_words = stopwords.words('english')
#         stemmer = SnowballStemmer('english')
#         for token in text.split():
#             if token not in stop_words:
#                 tokens.append(stemmer.stem(token))
#         self.text = " ".join(tokens)
#         self.data = pad_sequences(self.tokenizer.texts_to_sequences(self.text), maxlen=700)

#     def predict(self):
#         if self.data.any():
#             prediction = self.model.predict_classes(self.data)
#             if prediction[0] == 0:
#                 return "partially false"
#             elif prediction[0] == 1:
#                 return "false"
#             elif prediction[0] == 2:
#                 return "true"
#             elif prediction[0] == 3:
#                 return "other"


# class Three_Layer(Model):
#     def __init__(self, text):
#         super().__init__(text)

#     def download_dependencies(self):
#         try:
#             nltk.data.find('stopwords')
#             nltk.data.find('wordnet')
#             nltk.data.find('words')
#         except LookupError:
#             nltk.download('stopwords')
#             nltk.download('wordnet')
#             nltk.download('words')

#     def process_text(self):
#         self.clean_text = clean_text(self.text)

#         nlp = spacy.load('en_core_web_sm')
#         pos_tags_column = []

#         for text in self.text.split():
#             pos_tags = []
#             doc = nlp(text)
#             for token in doc:
#                 pos_tags.append(token.pos_)
#             all_pos_tags = ' '.join(pos_tags)
#             pos_tags_column.append(all_pos_tags)

#         self.POS_text = pos_tags_column

#         lexicon = Empath()
#         semantic = []
#         count = 0

#         d = lexicon.analyze(self.text.split(), normalize=False)
#         x = []
#         for key, value in d.items():
#             x.append(value)
#         x = np.asarray(x)
#         semantic.append(x)

#         self.semantic_text = semantic

#         categories = []
#         a = lexicon.analyze("")
#         for key, value in a.items():
#             categories.append(key)

#         sem = []
#         a = []
#         for j in range(len(semantic[0])):
#             for k in range(int(semantic[0][j])):
#                 a.append(categories[j])
#         b = " ".join(a)
#         sem.append(b)

#         self.semantics_text = sem

#         X_test_text = self.clean_text
#         X_test_POS = self.POS_text
#         X_test_sem = self.semantics_text

#         empty = ""
#         for pos in X_test_POS:
#             empty += pos + " "
#         X_test_POS = [empty]

#         loaded_tfidf = pickle.load(open(
#             '../../Desktop/AI-communication-module/models/Three_Layer/vectorizers/tfidf_pickle.sav', 'rb'))
#         loaded_pos = pickle.load(open(
#             '../../Desktop/AI-communication-module/models/Three_Layer/vectorizers/pos_pickle.sav', 'rb'))
#         loaded_sem = pickle.load(open(
#             '../../Desktop/AI-communication-module/models/Three_Layer/vectorizers/sem_pickle.sav', 'rb'))

#         tfidf_test = loaded_tfidf.transform([X_test_text])
#         pos_tfidf_test = loaded_pos.transform(X_test_POS)
#         sem_tfidf_test = loaded_sem.transform(X_test_sem)

#         text_w = 0.5 * 3
#         pos_w = 0.15 * 3
#         sem_w = 0.35 * 3

#         tfidf_test *= text_w
#         pos_tfidf_test *= pos_w
#         sem_tfidf_test *= sem_w

#         diff_n_rows = pos_tfidf_test.shape[0] - tfidf_test.shape[0]
#         d = sp.vstack((tfidf_test, sp.csr_matrix((diff_n_rows, tfidf_test.shape[1]))))
#         e = sp.hstack((pos_tfidf_test, d))

#         diff_n_rows = e.shape[0] - sem_tfidf_test.shape[0]
#         d = sp.vstack((sem_tfidf_test, sp.csr_matrix((diff_n_rows, sem_tfidf_test.shape[1]))))

#         self.X_test = sp.hstack((e, d))

#     def load_model(self):
#         self.loaded_model = pickle.load(open(
#             '../../Desktop/AI-communication-module/models/Three_Layer/three_layer_pickle.sav', 'rb'))

#     def predict(self):
#         self.load_model()
#         return self.loaded_model.predict(self.X_test)[0].lower()


# def parse_args():
#     parser = argparse.ArgumentParser(description='Fake News Classification')
#     parser.add_argument('text', metavar='text', type=str, nargs='+', help='Text to be classified')
#     args = parser.parse_args()
#     return ' '.join(args.text)


# def clean_text(text):
#     text = re.sub('[' + string.punctuation + ']', '', text)
#     text = re.sub(r"[-()\"#/@’;:<>{}`+=~|.!?,]", '', text)
#     text = text.lower().split()

#     stops = set(stopwords.words("english"))
#     text = [w for w in text if w not in stops]
#     text = " ".join(text)

#     text = re.sub(r'[^a-zA-Z\s]', u'', text, flags=re.UNICODE)

#     text = text.split()
#     l = WordNetLemmatizer()
#     lemmatized_words = [l.lemmatize(word) for word in text if len(word) > 2]
#     text = " ".join(lemmatized_words)

#     return text


# class BertModel(Model):
#     def __init__(self, model, text):
#         if torch.cuda.is_available():
#             self.device = torch.device('cuda')
#         else:
#             self.device = torch.device('cpu')
#         self.model = model
#         self.text = self.preprocess(text)
#         self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

#     def tokenize_map(self, sentence, labs='None'):
#         global labels
#         input_ids = []
#         attention_masks = []

#         for text in sentence:
#             #   (1) Tokenize the sentence.
#             #   (2) Prepend the `[CLS]` token to the start.
#             #   (3) Append the `[SEP]` token to the end.
#             #   (4) Map tokens to their IDs.
#             #   (5) Pad or truncate the sentence to `max_length`
#             #   (6) Create attention masks for [PAD] tokens.

#             encoded_dict = self.tokenizer.encode_plus(
#                 text,
#                 add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#                 truncation='longest_first',  # Activate and control truncation
#                 max_length=72,  # Max length according to our current RAM constraints.
#                 pad_to_max_length=True,  # Pad & truncate all sentences.
#                 return_attention_mask=True,  # Construct attn. masks.
#                 return_tensors='pt',  # Return pytorch tensors.
#             )

#             # Add the encoded sentence to the id list.
#             input_ids.append(encoded_dict['input_ids'])

#             # And its attention mask (differentiates padding from non-padding).
#             attention_masks.append(encoded_dict['attention_mask'])

#         # Convert the lists into tensors.
#         input_ids = torch.cat(input_ids, dim=0)
#         attention_masks = torch.cat(attention_masks, dim=0)

#         if labs != 'None':  # Setting this for using this definition for both train and test data so labels won't be a problem in our outputs.
#             labels = torch.tensor(labels)
#             return input_ids, attention_masks, labels
#         else:
#             return input_ids, attention_masks

#     def floatToString(self, text):
#         try:
#             float(text)
#             return str(text)
#         except ValueError:
#             return text

#     def line_removal(self, text):
#         all_list = [char for char in text if char not in "…-–—_©“”‘’•⋆»"]
#         clean_str = ''.join(all_list)
#         return clean_str

#     def punctuation_removal(self, text):
#         all_list = [char for char in text if char not in string.punctuation]
#         clean_str = ''.join(all_list)
#         return clean_str

#     def preprocess(self, text, stem=False):
#         stop_words = stopwords.words('english')
#         stemmer = SnowballStemmer('english')
#         lemmatizer = WordNetLemmatizer()
#         text = self.floatToString(text)
#         text = text.lower()
#         text = self.line_removal(text)
#         text = self.punctuation_removal(text)
#         text = re.sub(r'[!]+', '!', text)
#         text = re.sub(r'[?]+', '?', text)
#         text = re.sub(r'[.]+', '.', text)
#         text = re.sub(r"'", "", text)
#         text = re.sub('\s+', ' ', text).strip()
#         text = re.sub(r'&amp;?', r'and', text)
#         text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)
#         text = re.sub(r'[:"$%&\*+,-/:;<=>@\\^_`{|}~]+', '', text)
#         emoji_pattern = re.compile("["
#                                    u"\U0001F600-\U0001F64F"  # Emoticons
#                                    u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
#                                    u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
#                                    u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
#                                    u"\U00002702-\U000027B0"
#                                    u"\U000024C2-\U0001F251"
#                                    "]+", flags=re.UNICODE)
#         text = emoji_pattern.sub(r'EMOJI', text)

#         tokens = []
#         for token in text.split():
#             if token not in stop_words:
#                 tokens.append(lemmatizer.lemmatize(token))
#         return " ".join(tokens)

#     def predict(self):
#         test_input_ids_f, test_attention_masks_f = self.tokenize_map(self.text)
#         prediction_data_f = TensorDataset(test_input_ids_f, test_attention_masks_f)
#         prediction_sampler_f = SequentialSampler(prediction_data_f)
#         prediction_dataloader_f = DataLoader(prediction_data_f, sampler=prediction_sampler_f, batch_size=12)
#         self.model.eval()
#         predictions = []

#         for batch in prediction_dataloader_f:
#             batch = tuple(t.to(self.device) for t in batch)
#             b_input_ids, b_input_mask, = batch

#             with torch.no_grad():
#                 outputs = self.model(b_input_ids, token_type_ids=None,
#                                      attention_mask=b_input_mask)
#             logits = outputs[0]
#             logits = logits.detach().cpu().numpy()
#             predictions.append(logits)

#         print('    DONE.')

#         flat_predictions = [item for sublist in predictions for item in sublist]
#         flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

#         preds = pd.DataFrame()
#         preds["our rating"] = flat_predictions

#         pred1 = []
#         for i in range(len(preds)):
#             if preds['our rating'][i] == 0:
#                 preds['our rating'][i] = 'false'
#             if preds['our rating'][i] == 3:
#                 preds['our rating'][i] = 'true'
#             if preds['our rating'][i] == 1:
#                 preds['our rating'][i] = 'other'
#             if preds['our rating'][i] == 2:
#                 preds['our rating'][i] = 'partially false'

#         return preds['our rating'][0]


# class ROBERTA(torch.nn.Module, Model):
#     def __init__(self, text, dropout_rate=0.4):
#         super(ROBERTA, self).__init__()
#         # Model.__init__(text)
#         self.text = text
#         self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#         self.roberta = RobertaModel.from_pretrained('roberta-base', return_dict=False, num_labels=4)
#         self.d1 = torch.nn.Dropout(dropout_rate)
#         self.l1 = torch.nn.Linear(768, 64)
#         self.bn1 = torch.nn.LayerNorm(64)
#         self.d2 = torch.nn.Dropout(dropout_rate)
#         self.l2 = torch.nn.Linear(64, 4)

#         if torch.cuda.is_available():
#             self.device = torch.device('cuda:0')
#         else:
#             self.device = torch.device('cpu')

#     def download_dependencies(self):
#         try:
#             nltk.data.find('stopwords')
#             nltk.data.find('wordnet')
#         except LookupError:
#             nltk.download('stopwords')
#             nltk.download('wordnet')

#     def process_text(self):
#         lemmatizer = WordNetLemmatizer()
#         corpus = []

#         for i in range(len(self.text)):
#             review = re.sub('[^a-zA-Z]', ' ', self.text[i])
#             review = review.lower()
#             review = review.split()
#             review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
#             review = ' '.join(review)
#             corpus.append(review)
#         self.text = corpus

#     def load_checkpoint(self, path, model):
#         state_dict = torch.load(path, map_location=self.device)
#         model.load_state_dict(state_dict['model_state_dict'])

#         return state_dict['valid_loss']

#     def predict(self):
#         labels_output = []
#         labels = ['false', 'true', 'partially false', 'other']
#         roberta_encoded_dict = self.tokenizer.encode_plus(
#             self.text,  # Sentence to encode.
#             add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#             max_length=128,  # Pad & truncate all sentences.
#             pad_to_max_length=True,
#             return_attention_mask=True,  # Construct attn. masks.
#             return_tensors='pt',  # Return pytorch tensors.
#         )
#         roberta_encoded_dict = roberta_encoded_dict.to(self.device)
#         outputs = self(**roberta_encoded_dict)
#         labels_output.append(labels[outputs.argmax()])
#         return labels_output

#     def forward(self, input_ids, attention_mask):
#         _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         x = self.d1(x)
#         x = self.l1(x)
#         x = self.bn1(x)
#         x = torch.nn.Tanh()(x)
#         x = self.d2(x)
#         x = self.l2(x)

#         return x



if __name__ == '__main__':
    # text = parse_args()
    
    text = ''

    with open("input.txt", "r") as fin:
        text = ' '.join(fin.read())


    # Bi LSTM
    bilstm = BiLstm(text)
    bilstm.download_dependencies()
    bilstm.process_text()
    prediction_bilstm = bilstm.predict()

    # # Sentiment Analysis
    # sentiment = Sentiment(text)
    # sentiment.download_dependencies()
    # sentiment.process_text()
    # prediction_sentiment = sentiment.predict()

    # # Three-Layer
    # three_layer = Three_Layer(text)
    # three_layer.download_dependencies()
    # three_layer.process_text()
    # prediction_three_layer = three_layer.predict()

    # # Bert
    # try:
    #     loaded_model = pickle.load(open('models/Bert/finalized_model.sav', 'rb'))

    #     bert = BertModel(loaded_model, text)
    #     prediction_bert = bert.predict()
    #     print(prediction_bert)
    # except:
    #     prediction_bert = 'other'

    # # ROBERTA
    # roberta = ROBERTA(text)
    # roberta.to(roberta.device)
    # roberta.download_dependencies()
    # roberta.load_checkpoint('models/Roberta/model/model.pkl', roberta)
    # roberta.process_text()
    # prediction_roberta = roberta.predict()[0]

    # c = Counter([prediction_bilstm, prediction_sentiment, prediction_three_layer])
    # value, count = c.most_common()[0]
    value = prediction_bilstm
    if value:
        with open('scor.txt', 'w') as f:
            f.write(value)
    else:
        with open("scor.txt", "w") as f:
            f.write("none-dont-believe")