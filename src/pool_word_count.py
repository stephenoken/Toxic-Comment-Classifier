import pandas as pd
from multiprocessing import Pool
from collections import Counter
from functools import reduce
import pickle


def map_comments(word_2_int, comment):
    mapped_comments = [word_2_int[word] for word in comment]
    return mapped_comments


def count_words(comment):
    counter = Counter()
    counter.update(comment)
    return counter


def main():
    with Pool() as p:
        lazy_counts_list = p.imap(count_words, X)

        counts = reduce(lambda res, cur: res + cur, lazy_counts_list, Counter())
        word_counts = sorted(counts, key=counts.get, reverse=True)
        word_2_int = {word: ii for ii, word in enumerate(word_counts, 1)}
        x_encoded = pd.Series(map(lambda x: map_comments(word_2_int, x), X))
        print(x_encoded)
        pickle.dump(x_encoded, open('./pickles/mapped-comments2.p', 'wb'))


    # processed_data_set = train_test_split(X, y, test_size=0.33, random_state=42)
    # X_train_raw, X_test_raw, y_train, y_test = processed_data_set


if __name__ == '__main__':
    data = pd.read_csv('./data/train.csv')
    X = data['comment_text'].apply(lambda x: x.split())
    y = data[data.columns[2:]]
    main()
