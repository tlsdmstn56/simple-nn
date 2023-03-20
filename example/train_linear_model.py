import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from simplenn.layers import *


class Model:

    def __init__(self, layers, verbose=False):
        self.layers = layers

    def set_verbosity(self, verbose):
        for l in self.layers:
            if hasattr(l, 'verbose'):
                l.verbose=verbose

    def __call__(self, x:np.array):
        for l in self.layers:
            x = l(x)
        return x

    def backward(self, grad_output):
        for l in reversed(self.layers):
            grad_output = l.backward(grad_output)

def run_test(layers, epochs, num_data, batch, lr, pbar=True):
    def make_model(layers, ref=False):
        model = []
        for i, num_output in enumerate(layers):
            if i == 0:
                continue
            num_input = layers[i-1]
            layer = Linear(num_input, num_output)
            if ref:
                layer.w = np.random.uniform(-10, 10, size=(num_input, num_output))
                layer.b = np.random.uniform(-20, 20, size=(num_output, ))
            model.append(layer)
                
            if i + 1 < len(layers):
                model.append(Sigmoid())
        return Model(model)

    ref_model = make_model(layers, True) 
    model = make_model(layers, False) 
    model.set_verbosity(False)
    num_batch = (num_data + batch - 1) // batch

    fig, ax = plt.subplots()
    (ln,) = ax.plot([-1], [0], animated=True)
    ax.set_xlim(0, epochs)
    plt.show(block=False)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    fig.canvas.blit(fig.bbox)

    X = [np.random.normal(size=(1, layers[0], )) for _ in range(num_data)]
    Y = [ref_model(x) for x in X]   
    data_idx = list(range(len(X)))
    loss_history = []
    with tqdm(total=epochs, disable=not pbar) as t:
        for e in range(epochs):
            random.shuffle(data_idx)
            grads = []
            for i in range(0, len(data_idx), batch):
                x_batch = [X[j] for j in data_idx[i:(i+batch)]]
                y_ref_batch = [Y[j] for j in data_idx[i:(i+batch)]]
                batch_size = len(x_batch)
                x_batch = np.concatenate(x_batch).reshape(batch_size, -1)
                y_ref_batch = np.concatenate(y_ref_batch).reshape(batch_size, -1)
                y_pred_batch = model(x_batch)
                y_pred_batch = y_pred_batch.reshape(y_ref_batch.shape)
                grad = y_pred_batch - y_ref_batch
                grads.append(grad.copy())
                d = lr * grad
                model.backward(d)
            t.update()
            grads = np.concatenate(grads)
            loss = np.mean(grad**2)
            loss_history.append(loss)
            fig.canvas.restore_region(bg)
            ax.set_ylim(0, max(loss_history))
            ln.set_xdata(np.arange(len(loss_history)))
            ln.set_ydata(loss_history)
            ax.draw_artist(ln)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
            t.set_description(f'Epoch: {e:6} Loss: {float(loss):8.2f}')
    
run_test([256, 128, 128, 64, 64, 32, 16, 4, 1], 100, 10000, 100, 0.001)

