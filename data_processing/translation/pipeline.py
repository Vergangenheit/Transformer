from data_processing.translation import dm_translation as dm, tknize_translation as tk


def pipeline():
    raw_data_en, raw_data_fr_in, raw_data_fr_out = dm.split_raw_data()

    data_en, data_fr_in, data_fr_out = tk.tokenize(raw_data_en, raw_data_fr_in, raw_data_fr_out, True)

    dataset = tk.create_dataset(data_en, data_fr_in, data_fr_out)

    return data_en, data_fr_in, dataset


if __name__ == '__main__':
    dataset, data_en, data_fr_in = pipeline()
