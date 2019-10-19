import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from preprocessing import TextPreprocessor
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU,Dense, Activation,SpatialDropout1D
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.layers.core import Dense, Dropout, Activation

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

MAX_NB_WORDS = 100000
EMBEDDING_DIM = 128             
batch_size = 256                #batch大小
MAX_SEQUENCE_LENGTH = 50        #pad_sequence最大长度
VALIDATION_SPLIT = 0.1          #验证集比例

import os
path_prefix= os.path.abspath(os.path.join(os.getcwd(), "../"))
print(path_prefix)

pass

def load_dataset(datapath):
    data = pd.read_csv(datapath, lineterminator="\n")
    data["label"] = data.label.apply(lambda x: 0 if x < 0 else x)
    print(data.groupby('label').size().reset_index(name='counts'))
    return data

def build_data():
    process = TextPreprocessor(stopword_file=os.path.join(path_prefix, "data/stopwords/stopword_normal.txt"))
    train_data = load_dataset(os.path.join(path_prefix, "data/comment_trainset_2class.csv"))#.sample(frac=0.1)
    print("train shape: ", train_data.shape)

    X = train_data.CONTENT.apply(lambda x: process.process_line(x))
    y = np.array(train_data.label.tolist())

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    #Found 214909 unique tokens.

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(y))
    print(labels[:100])
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    data = data[indices]
    labels = labels[indices]

    n_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    X_train = data[:-n_validation_samples]
    y_train = labels[:-n_validation_samples]

    X_test = data[-n_validation_samples:]
    y_test = labels[-n_validation_samples:]
    return X_train, X_test, y_train, y_test, word_index, sequences, tokenizer

def build_testset(tokenizer):
    process = TextPreprocessor(stopword_file=os.path.join(path_prefix,"data/stopwords/stopword_normal.txt"))
    valid_data = load_dataset(os.path.join(path_prefix, "data/comment_testset_2class.csv"))#.sample(frac=0.01)
    print("valid_data shape: ", valid_data.shape)

    X = valid_data.CONTENT.apply(lambda x: process.process_line(x))
    y = np.array(valid_data.label.tolist())

    sequences = tokenizer.texts_to_sequences(X)

    X_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_val = to_categorical(np.asarray(y))
    print('Shape of data tensor:', X_val.shape)
    print('Shape of label tensor:', y_val.shape)
    return X_val, y_val, y


def build_embedding_matrix(word_index, sequences, embeddings_index={}):
    nb_words = min(MAX_NB_WORDS, len(word_index))
    #20000
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print(embedding_matrix.shape)
    #(20001, 100)
    return embedding_matrix, nb_words

def train_model(X_train, y_train, X_test, y_test, n_words, batch_size, n_class=2):
    embedding_layer = Embedding(n_words + 1, EMBEDDING_DIM,
                            # weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, dropout=0.2)
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(0.4))
    # model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))  # try using a GRU 
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2)) 
    model.add(Dense(n_class, activation='softmax'))

    # try using different parameters
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    model.fit(X_train, y_train, batch_size=batch_size, epochs=5, validation_data=(X_test, y_test))

    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:{}, accuracy:{}'.format(score, acc))
    return model

def performance(y_true, y_pred):
    accuracy = round(metrics.accuracy_score(y_true, y_pred)*100,3)
    confusion = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print("多模型融合预测accuracy：{}".format(accuracy))
    print("混淆矩阵：\n{}".format(confusion))
    print("预测结果：\n{}".format(report))
    return confusion, report

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, word_index, sequences, tokenizer = build_data()
    print("X_train: {}, X_test: {}".format(X_train.shape, X_test.shape))
    embedding_matrix, nb_words = build_embedding_matrix(word_index, sequences,embeddings_index={})
    print(embedding_matrix.shape, nb_words)
    X_val, y_val, y_true = build_testset(tokenizer)

    model = train_model(X_train, y_train, X_test, y_test, nb_words, batch_size)

    score, acc = model.evaluate(X_val, y_val,
                                batch_size=batch_size)
                            
    print('val score:', score)
    print('val accuracy:', acc)

    y_pred = model.predict_classes(X_val)
    y_pred_prob = model.predict_proba(X_val)
    performance(y_true, y_pred)
    # print(y_pred_prob)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_pred_prob[:,1])
    print("model auc score: {b}".format(b=auc))

    print("save model.")
    model.save(os.path.join(path_prefix, 'output/my_model-gru.h5'))

    import pickle

    # saving
    token_path = os.path.join(path_prefix, 'output/tokenizer-gru.pickle')
    with open(token_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # loading
    with open(token_path, 'rb') as handle:
        tokenizer = pickle.load(handle)