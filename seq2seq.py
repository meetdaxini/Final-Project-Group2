from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Model
import re
import numpy as np
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

if tf.config.list_physical_devices('GPU'):
    print('GPU is available. Using GPU.')
    device = '/GPU:0'
else:
    print('GPU is not available. Using CPU.')
    device = '/CPU:0'



# # Load the dataset
# dataset = load_dataset("multi_news")
#
# # Convert to Pandas DataFrames
# train_df = pd.DataFrame(dataset['train'])
# validation_df = pd.DataFrame(dataset['validation'])
# test_df = pd.DataFrame(dataset['test'])
#
# train_df.to_csv('train_data.csv', index=False)
# test_df.to_csv('test_data.csv', index=False)
# validation_df.to_csv('validation_data.csv', index=False)

data = pd.read_csv("test_data.csv")

# Replace story separator tag with newline character in the 'text' column
data['document'] = data['document'].astype(str).apply(lambda x: re.sub(r'(\|\|\|\|\||\n|\s+)', ' ', x))


# Remove leading "-" from summaries
data['summary'] = data['summary'].apply(lambda x: re.sub('^– ', '', x))
print(data['document'][0])
print(data['summary'][0])
# Get maximum lengths
max_text_length = max(data['document'].apply(lambda x: len(x.split())))
max_summary_length = max(data['summary'].apply(lambda x: len(x.split())))

print("Max text length:", max_text_length)
print("Max summary length:", max_summary_length)


# Define encoder
def build_encoder(input_dim, embedding_dim, lstm_units):
    encoder_inputs = Input(shape=(max_text_length,))
    encoder_embedding = Embedding(input_dim, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(lstm_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    return encoder_inputs, encoder_outputs, encoder_states


# Define decoder
def build_decoder(output_dim, embedding_dim, lstm_units):
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(output_dim, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding)
    decoder_dense = TimeDistributed(Dense(output_dim, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)
    return decoder_inputs, decoder_outputs


# Build the model
def build_seq2seq_model(input_dim, output_dim, embedding_dim, lstm_units, device):
  with tf.device(device):
    # Encoder
    encoder_inputs, encoder_outputs, encoder_states = build_encoder(input_dim, embedding_dim, lstm_units)

    # Decoder
    decoder_inputs, decoder_outputs = build_decoder(output_dim, embedding_dim, lstm_units)

    # Connect encoder and decoder
    # decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    # decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model





# Split data into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenization and preprocessing
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing and removing stopwords
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return tokens

# Apply tokenization and preprocess for text and summary
train_data['text_tokens'] = train_data['document'].apply(preprocess_text)
train_data['summary_tokens'] = train_data['summary'].apply(preprocess_text)

test_data['text_tokens'] = test_data['document'].apply(preprocess_text)
test_data['summary_tokens'] = test_data['summary'].apply(preprocess_text)




tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['text_tokens'])
text_vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token

print("Vocabulary Size:", text_vocab_size)


# Tokenize and pad sequences
# text
tokenizer = Tokenizer(num_words=text_vocab_size)
tokenizer.fit_on_texts(train_data['text_tokens'])

X_train_sequences = tokenizer.texts_to_sequences(train_data['text_tokens'])
X_test_sequences = tokenizer.texts_to_sequences(test_data['text_tokens'])

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_text_length, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_text_length, padding='post')



#summary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['summary_tokens'])
summary_vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token

print("Vocabulary Size:", summary_vocab_size)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=summary_vocab_size)
tokenizer.fit_on_texts(train_data['text_tokens'])

y_train_sequences = tokenizer.texts_to_sequences(train_data['summary_tokens'])
y_test_sequences = tokenizer.texts_to_sequences(test_data['summary_tokens'])

y_train_padded = pad_sequences(y_train_sequences, maxlen=max_summary_length, padding='post')
y_test_padded = pad_sequences(y_test_sequences, maxlen=max_summary_length, padding='post')


# Print shapes
print("X_train shape:", X_train_padded.shape)
print("X_test shape:", X_test_padded.shape)
print("y_train shape:", y_train_padded.shape)
print("y_test shape:", y_test_padded.shape)
type(y_test_padded)


embedding_dim = 256
lstm_units = 512


print(device)
model = build_seq2seq_model(text_vocab_size, summary_vocab_size, embedding_dim, lstm_units, device)
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit([X_train_padded, y_train_padded[:,:-1]], y_train_padded[:,1:],
                    epochs=10, batch_size=32, validation_data=([X_test_padded, y_test_padded[:,:-1]], y_test_padded[:,1:]))
