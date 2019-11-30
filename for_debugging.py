import pandas as pd
import numpy as np
import mxnet
from data_io_util import load_data, INPUT_SIGNAL_TYPES, LABELS
from mxnet_rnn_util import BaseRNNClassifier


if __name__ == '__main__':

    ctx = mxnet.cpu()
    model = BaseRNNClassifier(ctx)
    model.build_model(n_out=len(LABELS), rnn_size=64, n_layer=1)
    model.load_parameters('model.params', ignore_extra=True)

    test = ["data/har_data/test/%stest.txt" % signal for signal in INPUT_SIGNAL_TYPES]
    X_test = load_data(test)

    y_test_path = "data/har_data/test/y_test.txt"
    df = pd.read_csv(y_test_path, names=["labels"])
    y_test = np.asarray(list(df.to_dict()['labels'].values())).astype('float32')

    print(X_test.shape)
    print(y_test.shape)

    pred = model.predict([X_test], is_label_included=False)

    print(pred.shape)
    print(pred[y_test[3:2947] - pred].shape)
