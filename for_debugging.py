import pandas as pd
import numpy as np
import mxnet
from data_io_util import load_data, INPUT_SIGNAL_TYPES, LABELS
from mxnet_rnn_util import BaseRNNClassifier


def continue_training():
    ctx = mxnet.cpu()
    model = BaseRNNClassifier(ctx)
    model.build_model(n_out=len(LABELS), rnn_size=64, n_layer=1)
    model.load_parameters('model.params', ignore_extra=True)

    train = ["data/har_data/train/%strain.txt" % signal for signal in INPUT_SIGNAL_TYPES]
    test = ["data/har_data/test/%stest.txt" % signal for signal in INPUT_SIGNAL_TYPES]

    X_train = load_data(train)
    X_test = load_data(test)

    y_test_path = "data/har_data/test/y_test.txt"
    df = pd.read_csv(y_test_path, names=["labels"])
    y_test = np.asarray(list(df.to_dict()['labels'].values())).astype('float32')

    print(X_test.shape)
    print(y_test.shape)

    import pandas as pd

    y_train_path = "data/har_data/train/y_train.txt"

    df = pd.read_csv(y_train_path, names=["labels"])
    y_train = np.asarray(list(df.to_dict()['labels'].values())).astype('float32')

    ctx = mxnet.cpu()
    model = BaseRNNClassifier(ctx)
    model.build_model(n_out=len(LABELS), rnn_size=64, n_layer=1)
    model.compile_model()
    model.load_parameters('model.params', ignore_extra=True)

    train_loss, train_acc, test_acc = model.fit([X_train, y_train], [X_test, y_test], batch_size=32, epochs=25)
    model.save_parameters('model2.params')

    import matplotlib.pyplot as plt

    plt.plot(train_loss, label="loss")
    plt.plot(train_acc, label="train")
    plt.plot(test_acc, label="validation")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    continue_training()