import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Embedding, LSTM, Bidirectional, Dense
from keras.models import Input, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


############################################################
EMBED_SIZE = 300
NUM_WORDS = 222161
MAXLEN = 40
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


def load_and_preprocess_data(maxlen, seed=11):
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    # Tokenize the sentences
    tokenizer = Tokenizer()
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


def load_glove(word_index, num_words):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    embeddings_glove = dict(get_coefs(*line.split(" ")) for line in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_glove.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(num_words, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_glove.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("embedding_matrix shape:", embedding_matrix.shape)

    return embedding_matrix


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


def find_best_threshold(y_val, pred_y_val):
    best_threshold = 0.5
    best_f1_score = 0.0
    for thresh in np.arange(0.001, 0.999, 0.001):
        thresh = np.round(thresh, 3)
        score = f1_score(y_val, (pred_y_val > thresh).astype(int))
        if score > best_f1_score:
            best_threshold = thresh
            best_f1_score = score
    print("best_threshold: {}, best_f1_score: {}".format(best_threshold, best_f1_score))
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


def predict(model, test_X, best_threshold, val_best_f1_score):
    pred_test_y = model.predict(test_X, batch_size=512)
    pred_test_y = pred_test_y.reshape(-1)
    pred_test_y = np.where(pred_test_y > best_threshold, 1, 0)

    test_df = pd.read_csv("../input/test.csv")
    out_df = pd.DataFrame({"qid": test_df["qid"].values})
    out_df['prediction'] = pred_test_y
    out_df.to_csv("submission_{:.6f}.csv".format(val_best_f1_score), index=False)


def train_and_predict():
    train_X, test_X, train_y, word_index = load_and_preprocess_data(maxlen=MAXLEN, seed=SEED)
    embedding_matrix = load_glove(word_index, num_words=NUM_WORDS)

    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.05, random_state=SEED)

    model = model_lstm(embedding_matrix, maxlen=MAXLEN, embed_size=EMBED_SIZE, units=64)
    model = train(model, X_train, y_train, X_val, y_val, epochs=2)

    pred_y_val = model.predict(X_val, batch_size=512, verbose=0)
    best_threshold, best_f1_score = find_best_threshold(y_val, pred_y_val)

    predict(model, test_X, best_threshold, best_f1_score)


if __name__ == '__main__':
    train_and_predict()
