import pytest
import torch
import numpy as np

from layers import *

def test_conv1d():
    # example weight
    w = np.random.normal(size=(4, 3, 5))
    b = np.random.normal(size=(4, ))

    # example input
    x = torch.randn(2, 3, 10)

    # torch layer & weight update
    ref_layer = torch.nn.Conv1d(3, 4, 5)
    with torch.no_grad():
        ref_layer.weight.copy_(torch.tensor(w))
        ref_layer.bias.copy_(torch.tensor(b))
    
    # my impl layer & weight update
    layer = Conv1d(3, 4, 5)
    layer.w = w
    layer.b = b

    y_ref = ref_layer(x).detach().numpy()
    y_pred = layer(x.detach().numpy())

    np.testing.assert_allclose(y_ref, y_pred, rtol=1e-05)
    
