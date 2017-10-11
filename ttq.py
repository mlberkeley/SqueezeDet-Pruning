import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def Quantize(cls, t=0.05):
    """Quantizes a layer.

    Assumptions:
        class has weight and bias parameters: weight is not None, bias may be None
        pytorch API doesn't change "significantly"
    """
    class QuantizeMeta(type):
        def __new__(self, name, bases, dct):
            # Change base class to `cls` and prepend `TTQ` to the class name.
            dct.pop('__qualname__', None)
            return super().__new__(self, 'TTQ{}'.format(cls.__name__), (cls,), dct)

    class Quantized(metaclass=QuantizeMeta):
        """QuantizeMeta makes this class inherit from `cls` instead of `object`."""
        def __init__(self, *args, **kwargs):
            # hack: superclass constructor calls reset_parameters, but defer that call until we create our parameters
            self._super_ready = False
            super().__init__(*args, **kwargs)
            self._super_ready = True
            self.t = t
            self._override = {}
            self.W_p = nn.Parameter(torch.Tensor(1))
            self.W_n = nn.Parameter(torch.Tensor(1))
            self.reset_parameters()

        def __getattr__(self, name):
            """This needs to be defined in the class: changing this on an existing class doesn't work."""
            if name in self._override:
                return self._override[name]
            else:
                return super().__getattr__(name)

        def forward(self, *args, **kwargs):
            """Use quantized weights during forward pass:
            can't set self.weight and self.bias directly since those are pytorch "parameters",
            so do some __getattr__ hax instead.
            """
            if self.bias is not None:
                quantized_weight, quantized_bias = QuantizeWeights(self.t)(self.W_p, self.W_n, self.weight, self.bias)
            else:
                quantized_weight, = QuantizeWeights(self.t)(self.W_p, self.W_n, self.weight)
                quantized_bias = None
            self._override.update({'weight': quantized_weight, 'bias': quantized_bias})
            result = super().forward(*args, **kwargs)
            self._override.clear()
            return result

        def reset_parameters(self):
            if self._super_ready:
                super().reset_parameters()
                stdv = 1 / math.sqrt(self.weight.size(1))   # taken from nn.Linear.reset_parameters
                self.W_p.data.uniform_(0, stdv)
                self.W_n.data.uniform_(-stdv, 0)

    return Quantized

class QuantizeWeights(Function):
    def __init__(self, t):
        super(QuantizeWeights, self).__init__()
        self.t = t

    def forward(self, W_p, W_n, *weights):
        self.save_for_backward(W_p, W_n, *weights) # save_for_backward fails if you pass in non-argument tensors
        max_weight = max(weight.abs().max() for weight in weights)
        threshold = self.t * max_weight
        p_masks = [weight.gt(threshold).float() for weight in weights]
        n_masks = [weight.lt(-threshold).float() for weight in weights]
        quantized_weights = tuple(p_masks[i] * W_p + n_masks[i] * W_n for i in range(len(weights)))
        self._threshold = threshold
        self._p_masks = p_masks
        self._n_masks = n_masks
        return quantized_weights

    def backward(self, *grad_outputs):
        # grad_outputs are gradient of loss with respect to quantized weights
        W_p, W_n, *weights = self.saved_tensors
        threshold = self._threshold
        p_masks = self._p_masks
        n_masks = self._n_masks
        z_masks = [weight.abs().le(threshold).float() for weight in weights]
        quantized_weights_ = [p_masks[i] * W_p + n_masks[i] * -W_n for i in range(len(weights))] # note the extra minus
        out = [(quantized_weights_[i] + z_masks[i]) * grad_outputs[i] for i in range(len(weights))]
        W_p_grad = W_p.clone()
        W_n_grad = W_n.clone()
        W_p_grad[0] = sum((p_masks[i] * grad_outputs[i]).sum() for i in range(len(weights)))
        W_n_grad[0] = sum((n_masks[i] * grad_outputs[i]).sum() for i in range(len(weights)))
        return (W_p_grad, W_n_grad, *out)

class OldLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self._super_ready = False
        super().__init__(*args, **kwargs)
        self._super_ready = True
        self.W_p = nn.Parameter(torch.Tensor(1))
        self.W_n = nn.Parameter(torch.Tensor(1))
        self.t = 0.05
        self.reset_parameters()

    def reset_parameters(self):
        if self._super_ready:
            super().reset_parameters()
            stdv = 1 / math.sqrt(self.weight.size(1))
            self.W_p.data.uniform_(0, stdv)
            self.W_n.data.uniform_(-stdv, 0)

    def forward(self, input):
        quantized_weight, quantized_bias = QuantizeWeights(self.t)(self.W_p, self.W_n, self.weight, self.bias)
        return F.linear(input, quantized_weight, quantized_bias)
