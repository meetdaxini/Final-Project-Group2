import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Attention, Concatenate
from tensorflow.keras.models import Model
import re
import numpy as np
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import optimizers




if tf.config.list_physical_devices('GPU'):
    print('GPU is available. Using GPU.')
    device = '/GPU:0'
else:
    print('GPU is not available. Using CPU.')
    device = '/CPU:0'


data = pd.read_csv("test_data.csv")

# Replace story separator tag with newline character in the 'text' column
data['document'] = data['document'].astype(str).apply(lambda x: re.sub(r'(\|\|\|\|\||\n|\s+)', ' ', x))


# Remove leading "-" from summaries
data['summary'] = data['summary'].apply(lambda x: re.sub('^– ', '', x))


# Get maximum lengths
max_text_length = max(data['document'].apply(lambda x: len(x.split())))
max_summary_length = max(data['summary'].apply(lambda x: len(x.split())))

print("Max text length:", max_text_length)
print("Max summary length:", max_summary_length)




# Split data into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenization and preprocessing
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing and removing stopwords
    # tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return tokens


# Apply tokenization and preprocess for text and summary
train_data['text_tokens'] = train_data['document'].apply(preprocess_text)
train_data['summary_tokens'] = train_data['summary'].apply(preprocess_text)

test_data['text_tokens'] = test_data['document'].apply(preprocess_text)
test_data['summary_tokens'] = test_data['summary'].apply(preprocess_text)

#=======================================================================================================
def calculate_rare_words_coverage(thresh, tokens):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)

    count = 0
    total_count = 0
    frequency = 0
    total_frequency = 0

    for key, value in tokenizer.word_counts.items():
        total_count = total_count + 1
        total_frequency = total_frequency + value
        if value < thresh:
            count = count + 1
            frequency = frequency + value

    rare_word_percentage = (count / total_count) * 100
    rare_word_coverage = (frequency / total_frequency) * 100

    return count, total_count, rare_word_percentage, rare_word_coverage

#==========================================================================================================================

# text_tokens
thresh = 4
text_tokens = train_data['text_tokens']

count, total_count, rare_word_percentage, rare_word_coverage = calculate_rare_words_coverage(thresh, text_tokens)
print("% of rare words in vocabulary:", rare_word_percentage)
print("Total Coverage of rare words:", rare_word_coverage)
print(count)
print(total_count)


text_vocab_size = total_count - count + 1
print("Text Vocabulary Size:", text_vocab_size)


# Tokenize and pad sequences
# text
tokenizer = Tokenizer(num_words=text_vocab_size)
tokenizer.fit_on_texts(train_data['text_tokens'])

X_train_sequences = tokenizer.texts_to_sequences(train_data['text_tokens'])
X_test_sequences = tokenizer.texts_to_sequences(test_data['text_tokens'])

X_train_padded = pad_sequences(X_train_sequences, maxlen=5000, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=5000, padding='post')




# summary_tokens
thresh = 2
summary_tokens = train_data['summary_tokens']

count, total_count, rare_word_percentage, rare_word_coverage = calculate_rare_words_coverage(thresh, summary_tokens)
print("% of rare words in vocabulary:", rare_word_percentage)
print("Total Coverage of rare words:", rare_word_coverage)
print(count)
print(total_count)

summary_vocab_size = total_count - count + 1  # Add 1 for padding token
print("Summary Vocabulary Size:", summary_vocab_size)


# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=summary_vocab_size)
tokenizer.fit_on_texts(train_data['summary_tokens'])

y_train_sequences = tokenizer.texts_to_sequences(train_data['summary_tokens'])
y_test_sequences = tokenizer.texts_to_sequences(test_data['summary_tokens'])

y_train_padded = pad_sequences(y_train_sequences, maxlen=500, padding='post')
y_test_padded = pad_sequences(y_test_sequences, maxlen=500, padding='post')


# Print shapes
print("X_train shape:", X_train_padded.shape)
print("X_test shape:", X_test_padded.shape)
print("y_train shape:", y_train_padded.shape)
print("y_test shape:", y_test_padded.shape)




def build_seq2seq_model(text_vocab_size, summary_vocab_size, embedding_dim, lstm_units, device):
 with tf.device(device):
    # Encoder
    encoder_input = Input(shape=(None,))
    encoder_embedding = Embedding(text_vocab_size, embedding_dim)(encoder_input)
    encoder_lstm = LSTM(lstm_units, return_state=True)
    _, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding)

    # Decoder
    decoder_input = Input(shape=(None,))
    decoder_embedding = Embedding(summary_vocab_size, embedding_dim)(decoder_input)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[encoder_state_h, encoder_state_c])
    decoder_dense = Dense(summary_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_input, decoder_input], decoder_outputs)
    return model



embedding_dim = 128
lstm_units = 256


loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# train_accuracy = SparseCategoricalAccuracy()


print(device)
model = build_seq2seq_model(text_vocab_size, summary_vocab_size, embedding_dim, lstm_units, device)
# Compile the model
adam = optimizers.Adam(clipnorm=1.0)

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model

history = model.fit([X_train_padded, y_train_padded[:, :-1]], y_train_padded[:, 1:],
                    epochs=50, batch_size=16, validation_data=([X_test_padded, y_test_padded[:, :-1]], y_test_padded[:, 1:]))
