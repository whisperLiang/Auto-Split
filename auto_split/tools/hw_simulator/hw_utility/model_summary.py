import torch
from functools import partial
import numpy as np
import pandas as pd
# import distiller

def model_find_module_name(model, module_to_find):
    """Look up the name of a module in a model.

    Arguments:
        model: the model to search
        module_to_find: the module whose name we want to look up

    Returns:
        The module name (string) or None, if the module was not found.
    """
    for name, m in model.named_modules():
        if m == module_to_find:
            return name
    return None


def fc_visitor(self, input, output, df, model, memo):
    assert isinstance(self, torch.nn.Linear)
    if self in memo:
        return

    # Multiply-accumulate operations: MACs = #IFM * #OFM
    # Bias is ignored
    weights_vol = macs = volume(self.weight)
    attrs_type = 'FC'
    module_visitor(self, input, output, df, model, weights_vol, macs, attrs_type=attrs_type)


# Performance data collection  code follows from here down
def volume(tensor):
    """return the volume of a pytorch tensor"""
    if isinstance(tensor, torch.FloatTensor) or isinstance(tensor, torch.cuda.FloatTensor):
        return np.prod(tensor.shape)
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        return np.prod(tensor)
    raise ValueError

def size_to_str(torch_size):
    """Convert a pytorch Size object to a string"""
    assert isinstance(torch_size, torch.Size) or isinstance(torch_size, tuple) or isinstance(torch_size, list)
    return '('+(', ').join(['%d' % v for v in torch_size])+')'



def module_visitor(self, input, output, df, model, weights_vol, macs, kernel_size=None, attrs_type=None,
                   groups=None,stride=None,padding=None, bias=None):
    in_features_shape = input[0].size()
    out_features_shape = output.size()

    mod_name = model_find_module_name(model, self)
    # groups, stride, padding, bias

    df.loc[len(df.index)] = ([mod_name, self.__class__.__name__,
                              kernel_size if kernel_size is not None else '',
                              size_to_str(in_features_shape), volume(input[0]),
                              size_to_str(out_features_shape), volume(output),
                              int(weights_vol), int(macs),attrs_type,
                             groups if groups is not None else '',
                             stride if stride is not None else '',
                             padding if padding is not None else '',
                             bias if bias is not None else ''])

def conv_visitor(self, input, output, df, model, memo):

    assert isinstance(self, torch.nn.Conv2d)
    if self in memo:
        return
    weights_vol = volume(self.weight)

    # Multiply-accumulate operations: MACs = volume(OFM) * (#IFM * K^2) / #Groups
    # Bias is ignored
    macs = (volume(output) *
            (self.in_channels / self.groups * self.kernel_size[0] * self.kernel_size[1]))
    # attrs = 'k=' + '('+(', ').join(['%d' % v for v in self.kernel_size])+')'
    kernel_size = self.kernel_size
    groups = self.groups
    stride = self.stride
    padding = self.padding
    if self.bias is None:
        bias = False
    else:
        bias = True

    if self.groups >= self.in_channels:
        attrs_type = 'DW'
    elif self.kernel_size == (1, 1):
        attrs_type = 'PW'
    else:
        attrs_type = '3d'

    module_visitor(self, input, output, df, model, weights_vol, macs, kernel_size,attrs_type,groups,stride,padding,bias)


def model_performance_summary(model, dummy_input, batch_size=1):
    """Collect performance data"""

    def install_perf_collector(m):

        if isinstance(m, torch.nn.Conv2d):
            hook_handles.append(m.register_forward_hook(
                                    partial(conv_visitor, df=df, model=model, memo=memo)))
        elif isinstance(m, torch.nn.Linear):
            hook_handles.append(m.register_forward_hook(
                                    partial(fc_visitor, df=df, model=model, memo=memo)))

    df = pd.DataFrame(columns=['name', 'type', 'kernel_size', 'ifm', 'ifm_vol',
                               'ofm', 'ofm_vol', 'wgt_vol', 'mac','attr_type',
                               'groups', 'stride', 'padding', 'bias'
                               ])

    hook_handles = []
    memo = []

    # model = distiller.make_non_parallel_copy(model)
    model.apply(install_perf_collector)
    # Now run the forward path and collect the data
    dummy_input = dummy_input.to('cuda')
    model(dummy_input)
    # Unregister from the forward hooks
    for handle in hook_handles:
        handle.remove()

    return df