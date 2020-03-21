import config
import data_management as dm
import tokenize as tk


def pipeline():
    raw_data = dm.create_raw_data()
    raw_data_en, raw_data_fr_in, raw_data_fr_out = dm.split_raw_data(rawdata=raw_data)

    data_en, data_fr_in, data_fr_out = tk.tokenize(raw_data_en, raw_data_fr_in, raw_data_fr_out)

    dataset = tk.create_dataset(data_en, data_fr_in, data_fr_out)

    return dataset


if __name__ == '__main__':
    dataset = pipeline()
