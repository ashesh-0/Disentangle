from disentangle.core.enum import Enum
import numpy as np


class DataSplitType(Enum):
    All = 0
    Train = 1
    Val = 2
    Test = 3


def get_datasplit_tuples(val_fraction: float, test_fraction: float, total_size: int, ending_test: bool = False):
    if ending_test:
        # train => val => test
        train_fraction = 1 - test_fraction - val_fraction
        train = list(range(0, int(total_size * train_fraction)))
        val = list(range(train[-1] + 1, train[-1] + 1 + int(total_size * val_fraction)))
        test = list(range(val[-1] + 1, total_size))
        return train, val, test
    else:
        idx_list = np.random.RandomState(955).permutation(np.arange(total_size))
        # test => val => train
        test = list(range(0, int(total_size * test_fraction)))

        val = list(range(test[-1] + 1, test[-1] + 1 + int(total_size * val_fraction)))
        train = list(range(val[-1] + 1, total_size))

        return idx_list[train], idx_list[val], idx_list[test]


if __name__ == '__main__':
    train, val, test = get_datasplit_tuples(0.1, 0.1, 30)
    print(train)
    print(val)
    print(test)

    train, val, test = get_datasplit_tuples(0.1, 0.1, 30, ending_test=True)
    print(train)
    print(val)
    print(test)
