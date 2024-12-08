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

    vocab_size,total_classes,train_text_X,test_text_X,train_Y, test_Y = get_data_for_model_input()
    #to stack RNNs , return sequences must be true
    # model.add(SimpleRNN(latent_dim, recurrent_dropout=0.2, return_sequences=True, activation='tanh'))

    # Define the model
    model_clipping = Sequential([
        Embedding(vocab_size, embedding_dim, trainable=True),
        SimpleRNN(latent_dim, recurrent_dropout=0.2, return_sequences=False, activation='tanh'),
        Dense(total_classes, activation='softmax')
    ])

    model_clipping.summary()


    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)  # Clip gradients by norm
    model_clipping.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)
    
    model_clipping.fit(
            x=train_text_X,
            y=train_Y,
            validation_data=(test_text_X, test_Y),
            batch_size=64,
            epochs=20,  # Set a higher max epoch count, early stopping will halt it if needed
            callbacks=[early_stopping]
        )
    
    return model_clipping
    

    
def get_data_for_model_input():
    df = ingestData()
    df = get_cleaned_data(df)
    max_sentence_len,total_classes = get_max_seq_length(df)
    vocab_size,train_text_X,test_text_X,train_Y, test_Y = getTokenisedData()
    return vocab_size,total_classes,train_text_X,test_text_X,train_Y, test_Y


if __name__ == "__main__":
    trained_model = model_training()
    save_object(file_path=os.path.join("artifacts","model.pkl"),obj = trained_model)