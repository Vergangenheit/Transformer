from data_processing.translation import pipeline as pp
import positional_embedding as pe


def train():
    data_en, data_fr_in, dataset = pp.pipeline()
    pes = pe.build_pes(data_en, data_fr_in)


if __name__ == "__main__":
    train()
