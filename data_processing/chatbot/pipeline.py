import data_processing.chatbot.dm_chatbot as dm
import data_processing.chatbot.tknize_chatbot as tk


def pipeline():
    inputs, outputs = dm.load_conversations()
    tk.tokenize(inputs, outputs, True)
    questions, answers = tk.tokenize_and_filter(inputs, outputs)
    dataset = tk.create_dataset(questions, answers)

    return dataset


if __name__ == "__main__":
    dataset = pipeline()
