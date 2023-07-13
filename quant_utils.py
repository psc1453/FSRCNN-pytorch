import copy
import torch
from torch import nn
from models import FSRCNN
from typing import TypeVar

__all__ = ['quantize_tensor_with_original_scale', 'quantize_model_with_original_scale']

Module = TypeVar("Module", bound=nn.Module)


def quantize_tensor_with_original_scale(tensor_input: torch.Tensor, width: int) -> torch.Tensor:
    assert isinstance(tensor_input, torch.Tensor)
    assert isinstance(width, int)
    assert width > 0

    max_val = torch.max(tensor_input).item()
    min_val = torch.min(tensor_input).item()

    tensor_element_range = max_val - min_val

    if tensor_element_range == 0:
        return tensor_input
    else:
        steps = 2 ** width
        resolution = tensor_element_range / steps

        tensor_steps = tensor_input / resolution
        tensor_steps_int = torch.floor(tensor_steps)

        quantized_tensor = tensor_steps_int * resolution

        return quantized_tensor


def quantize_model_with_original_scale(model_input: nn.Module, weight_width: int, bias_weight: int) -> Module:
    model = copy.deepcopy(model_input)
    model_parameters = model.state_dict()
    for parameter_name in model_parameters:
        if 'weight' in parameter_name:
            model_parameters[parameter_name] = quantize_tensor_with_original_scale(
                tensor_input=model_parameters[parameter_name], width=bias_weight)
        elif 'bias' in parameter_name:
            model_parameters[parameter_name] = model_parameters[parameter_name] + 99
        else:
            raise KeyError('Unsupported state dict type found. (%s)' % parameter_name)
    model.load_state_dict(model_parameters)
    return model
