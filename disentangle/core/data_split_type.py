from disentangle.core.enum import Enum


class DataSplitType(Enum):
    All = 0
    Train = 1
    Val = 2
    Test = 3


def get_datasplit_tuples(val_fraction: float, test_fraction: float, total_size: int, starting_train: bool = True):
    if starting_train:
        # train => val => test
        train_fraction = 1 - test_fraction - val_fraction
        train = (0, int(total_size * train_fraction))
        val = (train[1], train[1] + int(total_size * val_fraction))
        test = (val[1], total_size)

    else:
        # test => val => train
        test = (0, int(total_size * test_fraction))
        val = (test[1], test[1] + int(total_size * val_fraction))
        train = (val[1], total_size)

    return train, val, test
