import numpy as np
from numba import njit

class Layer:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._saved_tensors = None

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def save_for_backward(self, *args):
        self._saved_tensors = args

    def get_saved_tensors(self):
        return self._saved_tensors


@njit
def _linear_call(x:np.array, w:np.array, b:np.array):
    num_input, num_output = w.shape
    B = x.shape[0]
    for i in range(B):
        x[i] = x[i].reshape((1, num_input)) @ w
    x += b.reshape((1, 1, num_output))
    return x

@njit
def _linear_backward(
        x:np.array, 
        grad_output:np.array, 
        w:np.array, 
        b:np.array):
    num_input, num_output = w.shape
    b -= grad_output.mean(axis=0).reshape(b.shape)
        
    dw = x.reshape((-1, num_input, 1))
    grad_output = grad_output.reshape((-1, 1,num_output))
    dw = dw @ grad_output

    w -= dw.mean(axis=0)
    dw = dw.mean(axis=2)
    assert dw.size == x.size
    return dw

class Linear(Layer):

    def __init__(self, num_input, num_output, verbose=False):
        super().__init__(verbose)
        self.num_input = num_input
        self.num_output = num_output
        self.w = np.random.normal(size=(1, num_input, num_output))
        self.b = np.random.normal(size=(1, num_output, 1))
        self._last_tensor = None
    
    def __call__(self, x:np.array):
        self._last_tensor = x.copy()
        y = _linear_call(x, self.w, self.b)
        self._print(self.num_input, self.num_output, 
                self._last_tensor.shape, '->', y.shape)
        return y

    def backward(self, grad_output:np.array):
        self._print(f'{grad_output.shape=}')
        return _linear_backward(
                self._last_tensor, grad_output, self.w, self.b)

class Conv1d(Layer):

    def __init__(self, 
            in_channel, 
            out_channel, 
            kernel_size, 
            verbose=False):
        super().__init__(verbose)
        self._last_tensor = None
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        w_shape = (out_channel, in_channel, kernel_size)
        self.w = np.zeros(w_shape)
        b_shape = (out_channel,)
        self.b = np.zeros(b_shape)

    def _get_output_shape(self, B, L):
        return (B, self.out_channel, L - self.kernel_size+1)

    def __call__(self, x:np.array):
        self._saved_tensor = x.copy()
        B, IN_C, L = x.shape
        assert IN_C == self.in_channel 
        y = np.zeros(self._get_output_shape(B, L))
        for b in range(B):
            for i in range(self.out_channel):
                for j in range(self.in_channel):
                    l = np.convolve(self.w[i, j, ::-1], x[b, j, :], 'valid')
                    y[b, i, :] += l

            # add bias
        y += self.b.reshape((1, self.out_channel, 1))

        return y



class ReLU(Layer):

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self._last_tensor = None

    def __call__(self, x):
        self._last_tensor = x.copy()
        return np.maximum(x, 0)


    def backward(self, grad_output):
        x = self._last_tensor
        
        grad = (np.sign(x) + 1) / 2.0
        
        return grad * grad_output.reshape(grad.shape)

class Sigmoid(Layer):

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self._last_tensor = None

    def __call__(self, x:np.array):
        self._last_tensor = x.copy()
        return self._sigmoid(x)

    @staticmethod
    @njit
    def _sigmoid(x:np.array):
        return 1/(1 + np.exp(-x))

    def backward(self, grad_output:np.array):
        x = self._last_tensor
        b = x.shape[0]

        x_sig = self._sigmoid(x)
        grad = x_sig * (1-x_sig)

        self._print(b)
        self._print(f'{grad_output.shape=}')
        self._print(f'{x.shape=}')
        return grad * grad_output.reshape(grad.shape)



