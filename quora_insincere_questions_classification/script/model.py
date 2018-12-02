import numpy as np
import pandas as pd
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Embedding, LSTM, GRU, Bidirectional, Dense, Layer
from keras.layers import SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Dropout
from keras.models import Input, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


############################################################
EMBED_SIZE = 300
NUM_WORDS = 150000
MAXLEN = 50
SEED = 11
############################################################


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    return 2*((precision_*recall_)/(precision_+recall_+K.epsilon()))


############################################################
def load_and_preprocess_data(maxlen, num_words, seed=11):
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    # Get the target values
    train_y = train_df['target'].values

    # shuffling the data
    np.random.seed(seed)
    train_idx = np.random.permutation(len(train_X))

    train_X = train_X[train_idx]
    train_y = train_y[train_idx]

    return train_X, test_X, train_y, tokenizer.word_index


def load_pretrained_embedding(word_index, num_words, embedding_type='glove', seed=11):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if embedding_type == 'glove':
        EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
        embeddings = dict(get_coefs(*line.split(" ")) for line in open(EMBEDDING_FILE))
    elif embedding_type == 'para':
        EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
        embeddings = dict(get_coefs(*line.split(" ")) for line in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'))
    else:
        EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
        embeddings = dict(get_coefs(*line.split(" ")) for line in open(EMBEDDING_FILE) if len(line) > 100)

    all_embs = np.stack(embeddings.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(num_words, len(word_index))
    np.random.seed(seed)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("embedding_matrix shape:", embedding_matrix.shape)

    return embedding_matrix


############################################################
class Attention(Layer):
    def __init__(self,
                 step_dim,
                 W_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.feature_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.feature_dim = input_shape[-1]

        self.W = self.add_weight('{}_W'.format(self.name),
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight('{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, feature_dim)),
                              K.reshape(self.W, (feature_dim, 1))),
                        (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.feature_dim


def model_lstm_gru_attention(embedding_matrix, maxlen, embed_size, units=64):
    num_words = len(embedding_matrix)
    input = Input(shape=(maxlen,))
    x = Embedding(num_words, embed_size, weights=[embedding_matrix], trainable=False)(input)
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    y = Bidirectional(GRU(units, return_sequences=True))(x)
    attention_1 = Attention(maxlen)(x)  # skip connection
    attention_2 = Attention(maxlen)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    concat = concatenate([attention_1, attention_2, avg_pool, max_pool])
    x = Dropout(0.2)(concat)
    x = Dense(units, activation='relu')(x)
    x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[precision, recall, f1])
    model.summary()
    return model


def model_lstm_attention(embedding_matrix, maxlen, embed_size, units=64):
    num_words = len(embedding_matrix)
    input = Input(shape=(maxlen,))
    x = Embedding(num_words, embed_size, weights=[embedding_matrix], trainable=False)(input)
    x = Bidirectional(LSTM(units*2, return_sequences=True))(x)
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(units, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[precision, recall, f1])
    model.summary()
    return model


def model_gru_attention(embedding_matrix, maxlen, embed_size, units=64):
    num_words = len(embedding_matrix)
    input = Input(shape=(maxlen,))
    x = Embedding(num_words, embed_size, weights=[embedding_matrix], trainable=False)(input)
    x = Bidirectional(GRU(units*2, return_sequences=True))(x)
    x = Bidirectional(GRU(units, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(units, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[precision, recall, f1])
    model.summary()
    return model


def model_lstm(embedding_matrix, maxlen, embed_size, units=64):
    num_words = len(embedding_matrix)
    input = Input(shape=(maxlen,))
    x = Embedding(num_words, embed_size, weights=[embedding_matrix], trainable=True)(input)
    x = Bidirectional(LSTM(units*2, return_sequences=True))(x)
    x = Bidirectional(LSTM(units, return_sequences=False))(x)
    x = Dense(units, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[precision, recall, f1])
    model.summary()
    return model


############################################################
def find_best_threshold(y_val, pred_y_val):
    best_threshold = 0.5
    best_f1_score = 0.0
    for thresh in np.arange(0.001, 0.999, 0.001):
        thresh = np.round(thresh, 3)
        score = f1_score(y_val, (pred_y_val > thresh).astype(int))
        if score > best_f1_score:
            best_threshold = thresh
            best_f1_score = score

    precision_ = precision_score(y_val, (pred_y_val > best_threshold).astype(int))
    recall_ = recall_score(y_val, (pred_y_val > best_threshold).astype(int))
    print("best_threshold: {}, best_f1_score: {}, precision: {}, recall: {}".format(
        best_threshold, best_f1_score, precision_, recall_))
    return best_threshold, best_f1_score


def train(model, X_train, y_train, X_val, y_val, epochs=1):
    model.fit(X_train,
              y_train,
              batch_size=512,
              epochs=epochs,
              verbose=1,
              callbacks=None,
              validation_data=(X_val, y_val),
              steps_per_epoch=None,
              validation_steps=None)
    return model


def predict(models, test_X, best_threshold):
    pred_test_y = []
    for i, model in enumerate(models):
        pred_test_y_ = model.predict(test_X, batch_size=512)
        pred_test_y_ = pred_test_y_.reshape(-1)
        pred_test_y.append(pred_test_y_)
    pred_test_y = np.mean(pred_test_y, axis=0)
    pred_test_y = np.where(pred_test_y > best_threshold, 1, 0)

    test_df = pd.read_csv("../input/test.csv")
    out_df = pd.DataFrame({"qid": test_df["qid"].values})
    out_df['prediction'] = pred_test_y
    out_df.to_csv("submission.csv", index=False)


def train_and_predict():
    train_X, test_X, train_y, word_index = load_and_preprocess_data(maxlen=MAXLEN, num_words=NUM_WORDS, seed=SEED)

    embedding_matrix_glove = load_pretrained_embedding(word_index, num_words=NUM_WORDS, embedding_type='glove')
    embedding_matrix_para = load_pretrained_embedding(word_index, num_words=NUM_WORDS, embedding_type='para')
    embedding_matrix = np.mean([embedding_matrix_glove, embedding_matrix_para], axis=0)

    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.05, random_state=SEED)
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)

    model_1 = model_lstm_attention(embedding_matrix, maxlen=MAXLEN, embed_size=EMBED_SIZE, units=64)
    model_2 = model_gru_attention(embedding_matrix, maxlen=MAXLEN, embed_size=EMBED_SIZE, units=64)
    model_3 = model_lstm_gru_attention(embedding_matrix, maxlen=MAXLEN, embed_size=EMBED_SIZE, units=64)

    models = [model_3]
    pred_y_val = []
    for i, model in enumerate(models):
        models[i] = train(model, X_train, y_train, X_val, y_val, epochs=5)
        pred_y_val_ = models[i].predict(X_val, batch_size=512, verbose=0)
        pred_y_val.append(pred_y_val_)
    pred_y_val = np.mean(pred_y_val, axis=0)

    best_threshold, best_f1_score = find_best_threshold(y_val, pred_y_val)
    predict(models, test_X, best_threshold)


train_and_predict()
