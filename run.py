from data_processing.chatbot import pipeline as pp
import train.train_chatbot as train


def run_train():
    questions, answers, dataset = pp.dataset_pipeline()
    train.train(dataset)


if __name__ == "__main__":
    run_train()
