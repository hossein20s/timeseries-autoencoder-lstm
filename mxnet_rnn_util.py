import mxnet as mx
import numpy as np
from mxnet import nd


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


class BaseRNNClassifier(mx.gluon.Block):
    '''
    Exensible RNN class with LSTM that can operate with MXNet NDArray iter or DataLoader.
    Includes fit() function to mimic the symbolic fit() function
    '''

    @classmethod
    def get_data(cls, batch, is_mx_iter_type, ctx, is_label_included=True):
        ''' get data and label from the iterator/dataloader '''
        if is_mx_iter_type:
            X = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx) if is_label_included else None
        else: # lif iter_type in ["numpy", "dataloader"]:
            X = mx.nd.array(batch[0]).as_in_context(ctx)
            y = mx.nd.array(batch[1]).as_in_context(ctx) if is_label_included else None
        #else:
        #    raise ValueError("iter_type must be mxiter or numpy")
        return X, y

    @classmethod
    def get_all_labels(cls, data_iterator, is_mx_iter_type):
        if is_mx_iter_type:
            pass
        else:
            return data_iterator._dataset._label

    def __init__(self, ctx):
        super(BaseRNNClassifier, self).__init__()
        self.ctx = ctx

    def build_model(self, n_out, rnn_size=128, n_layer=1):
        self.rnn_size = rnn_size
        self.n_layer = n_layer
        self.n_out = n_out

        # LSTM default; #TODO(Sunil): make this generic
        self.lstm = mx.gluon.rnn.LSTM(self.rnn_size, self.n_layer, layout='NTC')
        # self.lstm = mx.gluon.rnn.GRU(self.rnn_size, self.n_layer)
        self.output = mx.gluon.nn.Dense(self.n_out)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out[:, out.shape[1] - 1, :]
        out = self.output(out)
        return out, hidden

    def compile_model(self, loss=None, lr=3E-3):
        self.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        self.criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.loss = mx.gluon.loss.SoftmaxCrossEntropyLoss() if loss is None else loss
        self.lr = lr
        self.optimizer = mx.gluon.Trainer(self.collect_params(), 'adam',
                                          {'learning_rate': self.lr})

    def top_k_acc(self, data_iterator, is_mx_iter_type=True, top_k=3, batch_size=128):
        batch_pred_list = []
        true_labels = []
        init_state = mx.nd.zeros((self.n_layer, batch_size, self.rnn_size), self.ctx)
        hidden = [init_state] * 2
        for i, batch in enumerate(data_iterator):
            data, label = BaseRNNClassifier.get_data(batch, is_mx_iter_type, self.ctx)
            batch_pred = self.forward(data, hidden)
            # batch_pred = mx.nd.argmax(batch_pred, axis=1)
            batch_pred_list.append(batch_pred.asnumpy())
            true_labels.append(label)
        y = np.vstack(batch_pred_list)
        true_labels = np.vstack(true_labels)
        argsorted_y = np.argsort(y)[:, -top_k:]
        return np.asarray(np.any(argsorted_y.T == true_labels, axis=0).mean(dtype='f'))

    def evaluate_accuracy(self, data_iterator, metric='acc', is_mx_iter_type=True, batch_size=128):
        met = mx.metric.Accuracy()
        init_state = mx.nd.zeros((self.n_layer, batch_size, self.rnn_size), self.ctx)
        hidden = [init_state] * 2
        for i, batch in enumerate(data_iterator):
            data, label = BaseRNNClassifier.get_data(batch, is_mx_iter_type, self.ctx)
            # Lets do a forward pass only!
            output, hidden = self.forward(data, hidden)
            preds = mx.nd.argmax(output, axis=1)
            met.update(labels=label, preds=preds)

        # if self.all_labels is None:
        #    self.all_labels = BaseRNNClassifier.get_all_labels(data_iterator, iter_type)
        # preds = self.predict(data_iterator, iter_type=iter_type, batch_size=batch_size)
        # met.update(labels=mx.nd.array(self.all_labels[:len(preds)]), preds=preds)

        return met.get()


    def fit(self, train_data, test_data, epochs, batch_size, verbose=True):
        '''
        @train_data:  can be of type list of Numpy array, DataLoader, MXNet NDArray Iter
        '''

        moving_loss = 0.

        train_loss = []
        train_acc = []
        test_acc = []

        print("Data type:", type(train_data), type(test_data), type(train_data[0]))

        train_iter, total_batches, is_mx_iter_type = BaseRNNClassifier.to_nd_array_iter(train_data, batch_size)
        test_iter, _, _ = BaseRNNClassifier.to_nd_array_iter(test_data, batch_size)

        print("Data type:", type(train_data), type(test_data), is_mx_iter_type)
        print("Sizes", self.n_layer, batch_size, self.rnn_size, self.ctx)

        for e in range(epochs):
            # print self.lstm.collect_params()

            # reset iterators if of MXNet Itertype
            if is_mx_iter_type:
                train_iter.reset()
                test_iter.reset()

            init_state = mx.nd.zeros((self.n_layer, batch_size, self.rnn_size), self.ctx)
            hidden = [init_state] * 2
            # hidden = self.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=self.ctx)
            yhat = []
            for i, batch in enumerate(train_iter):
                data, label = BaseRNNClassifier.get_data(batch, is_mx_iter_type, self.ctx)
                # print "Data Shapes:", data.shape, label.shape
                hidden = detach(hidden)
                with mx.autograd.record(train_mode=True):
                    preds, hidden = self.forward(data, hidden)
                    # print preds[0].shape, hidden[0].shape, label.shape
                    loss = self.loss(preds, label)
                    yhat.extend(preds)
                loss.backward()
                self.optimizer.step(batch_size)
                preds = mx.nd.argmax(preds, axis=1)

                batch_acc = mx.nd.mean(preds == label).asscalar()

                if i == 0:
                    moving_loss = nd.mean(loss).asscalar()
                else:
                    moving_loss = .99 * moving_loss + .01 * mx.nd.mean(loss).asscalar()

                if verbose and i % 100 == 0:
                    print('[Epoch {}] [Batch {}/{}] Loss: {:.5f}, Batch acc: {:.5f}'.format(
                        e, i, total_batches, moving_loss, batch_acc))

            train_loss.append(moving_loss)

            t_acc = self.evaluate_accuracy(data_iterator=train_iter, is_mx_iter_type=is_mx_iter_type, batch_size=batch_size)
            train_acc.append(t_acc[1])

            tst_acc = self.evaluate_accuracy(data_iterator=test_iter, is_mx_iter_type=is_mx_iter_type, batch_size=batch_size)
            test_acc.append(tst_acc[1])

            print("Epoch %s. Loss: %.5f Train Acc: %s Test Acc: %s" % (e, moving_loss, t_acc, tst_acc))
        return train_loss, train_acc, test_acc

    @staticmethod
    def to_nd_array_iter(data, batch_size=None, is_label_included=True):
        data_iter = None
        total_batches = None
        is_mx_iter_type = False
        # Can take MX NDArrayIter, or DataLoader
        if isinstance(data, mx.io.NDArrayIter):
            data_iter = data
            is_mx_iter_type = True
            # total_batches = train_iter.num_data // train_iter.batch_size

        elif isinstance(data, list):
            if isinstance(data[0], np.ndarray) and (~is_label_included or isinstance(data[1], np.ndarray)):
                tX = np.asarray(data[0]).astype('float32')
                ty = np.asarray(data[1]).astype('float32') if is_label_included else None

                total_batches = tX.shape[0] // batch_size
                data_iter = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(tX, ty) if is_label_included else mx.gluon.data.ArrayDataset(tX, tX),
                                                     batch_size=batch_size, shuffle=False, last_batch='discard')

        elif isinstance(data, mx.gluon.data.dataloader.DataLoader):
            data_iter = data
            total_batches = len(data_iter)
        else:
            raise ValueError("pass mxnet ndarray or numpy array as [data, label]")
        return data_iter, total_batches, is_mx_iter_type

    def predict(self, data, batch_size=128, is_label_included=True):
        batch_pred_list = []
        init_state = mx.nd.zeros((self.n_layer, batch_size, self.rnn_size), self.ctx)
        hidden = [init_state] * 2
        print("Data type:", type(data), type(data[0]))
        data, _, is_mx_iter_type = BaseRNNClassifier.to_nd_array_iter(data, batch_size,
                                                                      is_label_included=is_label_included)
        print("Data type after:", type(data))
        print("Sizes", self.n_layer, batch_size, self.rnn_size, self.ctx)
        for i, batch in enumerate(data):
            data, _ = BaseRNNClassifier.get_data(batch, is_mx_iter_type=False, ctx=self.ctx, is_label_included=is_label_included)
            # print(data.shape)
            output, hidden = self.forward(data, hidden)
            batch_pred_list.append(output.asnumpy())
        # return np.vstack(batch_pred_list)
        print(len(batch_pred_list))
        return np.argmax(np.vstack(batch_pred_list), 1)


