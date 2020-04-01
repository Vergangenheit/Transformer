import predict
import os


def run_pred():
    sentence = input()
    answer = predict.predict(sentence)
    print(answer)


if __name__ == "__main__":
    run_pred()
