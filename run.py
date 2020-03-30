from data_processing.chatbot import pipeline as pp
import positional_embedding as pe
import train.train_chatbot as train


def run_train():
    questions, answers, dataset = pp.pipeline()
    pes = pe.build_pes(questions, answers)
    train.train(dataset, pes)


if __name__ == "__main__":
    run_train()
