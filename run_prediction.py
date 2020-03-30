import predict
import os


def run_pred():
    sentence = input("Enter question")
    generated_answer = predict.predict(sentence)
    print(generated_answer)


if __name__ == "__main__":
    run_pred()
