import pickle
from multiprocessing import Pool
import numpy as np

def predict(clfs, x):
    return [clf.predict_proba(x)[0][1] for clf in clfs]

def predict_all(batch):
    return [predict(classifiers, x) for x in batch]

def create_batch_generator(x, batch_size=64):
    n_batches = len(x)
    print(n_batches)
    x = x[: n_batches * batch_size]
    print(x)
    for ii in range(0, len(x), batch_size):
        print(ii)
        yield x[ii: ii + batch_size]

if __name__ == '__main__':
    with open('./pickles/naive-classifiers.p', 'rb') as f1, open('./pickles/x-test.p', 'rb') as f2:
        classifiers = pickle.load(f1)
        X_test = pickle.load(f2)
        print("Loaded files")
        with Pool() as p:
            batch_size = .2
            batches = create_batch_generator(X_test[:5])
            lazy_list = p.imap(predict_all, X_test)
            #
            res = np.array([x for x in lazy_list])
            pickle.dump(res, open('./pickles/result.p', 'wb'))
