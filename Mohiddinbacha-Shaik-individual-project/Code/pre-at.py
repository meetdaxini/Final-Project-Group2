import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Attention, Concatenate, Dropout
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
from tensorflow.keras.optimizers import Adam




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
tokenizer_text = Tokenizer(num_words=text_vocab_size)
tokenizer_text.fit_on_texts(train_data['text_tokens'])

X_train_sequences = tokenizer_text.texts_to_sequences(train_data['text_tokens'])
X_test_sequences = tokenizer_text.texts_to_sequences(test_data['text_tokens'])

X_train_padded = pad_sequences(X_train_sequences, maxlen=1500, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=1500, padding='post')




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
tokenizer_summary = Tokenizer(num_words=summary_vocab_size)
tokenizer_summary.fit_on_texts(train_data['summary_tokens'])

y_train_sequences = tokenizer_summary.texts_to_sequences(train_data['summary_tokens'])
y_test_sequences = tokenizer_summary.texts_to_sequences(test_data['summary_tokens'])

y_train_padded = pad_sequences(y_train_sequences, maxlen=200, padding='post')
y_test_padded = pad_sequences(y_test_sequences, maxlen=200, padding='post')


# Print shapes
print("X_train shape:", X_train_padded.shape)
print("X_test shape:", X_test_padded.shape)
print("y_train shape:", y_train_padded.shape)
print("y_test shape:", y_test_padded.shape)




def build_seq2seq_model(text_vocab_size, summary_vocab_size, embedding_dim, lstm_units, num_layers, dropout_rate, device):
    with tf.device(device):
        # Encoder
        encoder_input = Input(shape=(None,))
        encoder_embedding = Embedding(text_vocab_size, embedding_dim)(encoder_input)

        encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, dropout=dropout_rate)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_input = Input(shape=(None,))
        decoder_embedding = Embedding(summary_vocab_size, embedding_dim)(decoder_input)

        decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, dropout=dropout_rate)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        # Attention mechanism
        attention = Attention()
        attention_output = attention([decoder_outputs, encoder_outputs])
        concat_output = Concatenate(axis=-1)([decoder_outputs, attention_output])

        # Decoder Dense layer
        decoder_dense = Dense(summary_vocab_size, activation='softmax')
        decoder_final_output = decoder_dense(concat_output)

        # Full model for training
        model = Model([encoder_input, decoder_input], decoder_final_output)

        # Define the encoder model
        encoder_model = Model(encoder_input, [encoder_outputs,encoder_states])

        # Inputs for inference
        decoder_state_input_h = Input(shape=(lstm_units,))
        decoder_state_input_c = Input(shape=(lstm_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_embedding2 = Embedding(summary_vocab_size, embedding_dim)(decoder_input)
        # Using the same LSTM layer as in training
        inf_decoder_outputs, inf_state_h, inf_state_c = decoder_lstm(decoder_embedding2, initial_state=decoder_states_inputs)
        inf_attention_output = attention([inf_decoder_outputs, encoder_outputs])
        inf_concat_output = Concatenate(axis=-1)([inf_decoder_outputs, inf_attention_output])
        inf_decoder_final_output = decoder_dense(inf_concat_output)

        # Construct the inference decoder model
        decoder_model = Model(
            [decoder_input] + [encoder_outputs,decoder_states_inputs],
            [inf_decoder_final_output, inf_state_h, inf_state_c]
        )

        return model, encoder_model, decoder_model


embedding_dim = 256
lstm_units = 128
num_layers = 2
dropout_rate = 0.5
learning_rate = 0.001


print(device)
model, encoder_model, decoder_model = build_seq2seq_model(text_vocab_size, summary_vocab_size, embedding_dim, lstm_units, num_layers, dropout_rate, device)


# Compile the model
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model

history = model.fit([X_train_padded, y_train_padded], y_train_padded,
                    epochs=5, batch_size=16, validation_data=([X_test_padded, y_test_padded], y_test_padded))


reverse_target_word_index = tokenizer_summary.index_word
reverse_source_word_index = tokenizer_text.index_word
target_word_index = tokenizer_summary.word_index


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, [e_h, e_c] = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    stop_condition = False
    decoded_sentence = ''
    max_summary_length = 200  # Maximum length for summary
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        if sampled_token_index == 0:
            sampled_token = 'UNK'  # Handle unknown token
        else:
            sampled_token = reverse_target_word_index.get(sampled_token_index, 'UNK')

        decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if len(decoded_sentence.split()) >= (max_summary_length - 1):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


# Convert indices back to text
def indices_to_text(indices, tokenizer):
    return tokenizer.sequences_to_texts(indices)


# Print actual and predicted text for the first 5 samples
for i in range(5):
    input_text = indices_to_text([X_test_padded[i]], tokenizer_text)[0]
    actual_summary = indices_to_text([y_test_padded[i]], tokenizer_summary)[0]

    print(f"Sample {i + 1}:")
    print("Input Text:", input_text)
    print("Actual Summary:", actual_summary)
    print("Predicted summary:", decode_sequence(X_test_padded[i].reshape(1,1500)))
    print()

