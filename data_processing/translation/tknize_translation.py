import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import config_trans
import pickle


def tokenize(raw_data_en, raw_data_fr_in, raw_data_fr_out, save_en_tok=True, save_fr_tok=True):
    en_tokenizer = Tokenizer(filters='')
    en_tokenizer.fit_on_texts(raw_data_en)
    data_en = en_tokenizer.texts_to_sequences(raw_data_en)
    data_en = pad_sequences(data_en, padding='post')
    #save tokenizer
    if save_en_tok:
        with open(config_trans.EN_TOKENIZER, 'wb') as f:
            pickle.dump(en_tokenizer, f)
    fr_tokenizer = Tokenizer(filters='')
    fr_tokenizer.fit_on_texts(raw_data_fr_in)
    fr_tokenizer.fit_on_texts(raw_data_fr_out)
    data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
    data_fr_in = pad_sequences(data_fr_in, padding='post')

    data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
    data_fr_out = pad_sequences(data_fr_out, padding='post')
    # save fr_tokenizer
    if save_fr_tok:
        with open(config_trans.FR_TOKENIZER, 'wb') as f:
            pickle.dump(fr_tokenizer, f)

    return data_en, data_fr_in, data_fr_out


def create_dataset(data_en, data_fr_in, data_fr_out):
    dataset = tensorflow.data.Dataset.from_tensor_slices(
        (data_en, data_fr_in, data_fr_out))
    dataset = dataset.shuffle(len(data_en)).batch(config_trans.BATCH_SIZE)

    return dataset
