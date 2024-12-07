from library import *


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def ingestData():
    df = pd.read_csv("/Users/prupro/Desktop/Github/NewsClassRNN/artifacts/bbc-news-data.csv", sep="\t")
    df = df[((~df.title.isnull()) & (~df.content.isnull()))].reset_index(drop=True)
    print(f'Data shape and size is : {df.shape} and {df.shape[0]}')
    return df

def data_cleaning(text):

    # Lower the words in the sentence
    cleaned = text.lower()
    # Replace the full stop with a full stop and space
    cleaned = cleaned.replace(".", ". ")
    # Remove the stop words : optional pre-processing step
    tokens = [word for word in cleaned.split() if not word in stop_words]
    # Remove the punctuations
    tokens = [tok.translate(str.maketrans(' ', ' ', string.punctuation)) for tok in tokens]
    # Joining the tokens back to form the sentence
    cleaned = " ".join(tokens)
    # Remove any extra spaces
    cleaned = cleaned.strip()

    return cleaned

def get_max_seq_length(df):
    max_sentence_len = df['title'].str.split(" ").str.len().max()
    total_classes = df.category.nunique()
    print(f"Maximum sequence length: {max_sentence_len}")
    print(f"Total classes: {total_classes}")
    return max_sentence_len,total_classes

def get_cleaned_data(df):
    tqdm.pandas()
    # Apply the data_cleaning function to the 'title' column
    df['title'] = df['title'].progress_apply(data_cleaning)
    print(df.head())
    return df

def splitData(df):
    np.random.seed(100)
    train_X, test_X, train_Y, test_Y = train_test_split(df['title'],
                                                        df['category'],
                                                        test_size=0.2,
                                                        random_state=100)
    train_X = train_X.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    train_Y = train_Y.reset_index(drop=True)
    test_Y = test_Y.reset_index(drop=True)
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    train_Y,test_Y = encodeLabels(train_Y,test_Y)
    validation = test_Y.argmax(axis=1)
    return train_X,test_X,train_Y,test_Y,validation

def encodeLabels(train_Y,test_Y):
    train_Y = pd.get_dummies(train_Y).values
    test_Y = pd.get_dummies(test_Y).values
    return train_Y,test_Y

def getTokenisedData():
    df = ingestData()
    df = get_cleaned_data(df)
    max_sentence_len, total_classes = get_max_seq_length(df)
    train_X, test_X, _, _, _ = splitData(df)

    # Create and adapt the TextVectorization layer
    text_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=None,                     # No limit on vocabulary size
        output_mode='int',                   # Return integer indices
        output_sequence_length=int(max_sentence_len)  # Pad/truncate sequences to max_sentence_len
    )
    
    # Adapt the TextVectorization layer on the training data
    text_vectorizer.adapt(train_X)

    # Apply the vectorization to train and test data
    train_text_X = text_vectorizer(train_X)
    test_text_X = text_vectorizer(train_X)

    # Get the vocabulary size
    vocab_size = len(text_vectorizer.get_vocabulary())  # No need to add 1 manually

    print("Overall text vocab size:", vocab_size)
    return vocab_size,train_text_X,test_text_X

if __name__ == "__main__":
    df = ingestData()
    df = get_cleaned_data(df)
    train_X,test_X,train_Y,test_Y,validation = splitData(df)
    print(train_X[:5],test_X[:5],train_Y[:5])
