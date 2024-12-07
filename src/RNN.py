from library import *
from data_processing import *

#these are the hyper parameters
#h0 and EM dimensions(vxembedding_dim=> v is total number of words in the entire corpus)
latent_dim=50
embedding_dim=100

def model_training():
    seed=56
    tf.random.set_seed(seed)
    np.random.seed(seed)

    vocab_size,total_classes,train_text_X,test_text_X = get_data_for_model_input()
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, trainable=True))
    #to stack RNNs , return sequences must be true
    # model.add(SimpleRNN(latent_dim, recurrent_dropout=0.2, return_sequences=True, activation='tanh'))
    model.add(SimpleRNN(latent_dim, recurrent_dropout=0.2, return_sequences=False, activation='tanh'))
    model.add(Dense(total_classes, activation='softmax'))
    model.summary()

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_acc',
                                mode='max',
                                verbose=1,
                                patience=5)

    model.fit(x=train_text_X, y=train_Y,
            validation_data=(test_text_X, test_Y),
            batch_size=64,
            epochs=10,
            callbacks=[early_stopping])

def get_data_for_model_input():
    df = ingestData()
    df = get_cleaned_data(df)
    max_sentence_len,total_classes = get_max_seq_length(df)
    vocab_size,train_text_X,test_text_X = getTokenisedData()
    return vocab_size,total_classes,train_text_X,test_text_X


if __name__ == "__main__":
    model_training()