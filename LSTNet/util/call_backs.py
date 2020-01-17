from tensorflow.keras.callbacks import Callback, LambdaCallback

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch, logs: print(batch))



