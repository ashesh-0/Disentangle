from disentangle.core.enum import Enum
import numpy as np


class DataSplitType(Enum):
    All = 0
    Train = 1
    Val = 2
    Test = 3


def split_in_half(s, e):
    n = e - s
    s1 = list(np.arange(n // 2))
    s2 = list(np.arange(n // 2, n))
    return [x + s for x in s1], [x + s for x in s2]


def get_datasplit_tuples(val_fraction: float, test_fraction: float, total_size: int, starting_test: bool = False):
    if starting_test:
        # test => val => train
        test = list(range(0, int(total_size * test_fraction)))

        val = list(range(test[-1] + 1, test[-1] + 1 + int(total_size * val_fraction)))
        train = list(range(val[-1] + 1, total_size))
    else:

        # {test,val}=> train
        test_val_size = int((val_fraction + test_fraction) * total_size)
        train = list(range(test_val_size, total_size))

        # Split the test and validation in chunks.
        chunksize = 3

        nchunks = test_val_size // chunksize

        test = []
        val = []
        s = 0
        for i in range(nchunks):
            if i % 2 == 0:
                val += list(np.arange(s, s + chunksize))
            else:
                test += list(np.arange(s, s + chunksize))
            s += chunksize

        if i % 2 == 0:
            test += list(np.arange(s, test_val_size))
        else:
            p1, p2 = split_in_half(s, test_val_size)
            test += p1
            val += p2

    import pdb
    pdb.set_trace()
    return train, val, test


if __name__ == '__main__':
    train, val, test = get_datasplit_tuples(0.1, 0.1, 30)
    print(train)
    print(val)
    print(test)

    train, val, test = get_datasplit_tuples(0.1, 0.1, 30, starting_test=True)
    print(train)
    print(val)
    print(test)
