import data_processing.dm_chatbot as dm
import data_processing.tknize_chatbot as tk


def dataset_pipeline():
    questions, answers = dm.load_conversations()
    tk.tokenize(questions, answers, True)
    tk_questions, tk_answers = tk.tokenize_and_filter(questions, answers)
    dataset = tk.create_dataset(tk_questions, tk_answers)

    return questions, answers, dataset


if __name__ == "__main__":
    dataset = dataset_pipeline()
