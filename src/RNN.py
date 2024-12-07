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

    vocab_size,total_classes = get_data_for_model_input()
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, trainable=True))
    model.add(SimpleRNN(latent_dim, recurrent_dropout=0.2, return_sequences=False, activation='tanh'))
    model.add(Dense(total_classes, activation='softmax'))
    (model.summary())

def get_data_for_model_input():
    df = ingestData()
    df = get_cleaned_data(df)
    max_sentence_len,total_classes = get_max_seq_length(df)
    vocab_size = getTokenisedData()
    return vocab_size,total_classes


if __name__ == "__main__":
    model_training()