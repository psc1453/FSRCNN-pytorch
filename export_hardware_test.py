import torch
from torch import nn, fx
from PIL import Image

from models import FSRCNN
from utils import preprocess

from ModelModifier.modifier.utils import get_node_input, get_node_output, set_node_input

device = 'cpu'
SCALE = 3
INPUT_WIDTH = 8
WEIGHT_WIDTH = 8
BIAS_WIDTH = 18


class HardwareModel(nn.Module):
    def __init__(self):
        super(HardwareModel, self).__init__()
        self.conv1 = nn.Conv2d(12, 12, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return x


def export_parameters():
    model = FSRCNN(scale_factor=SCALE).eval()
    model.load_state_dict(torch.load('data/checkpoints/fsrcnn_x3.pth', map_location=device))
    parameter_dict = dict(model.state_dict())
    selected_weight = parameter_dict['mid_part.8.weight']
    selected_bias = parameter_dict['mid_part.8.bias']
    save_data(selected_weight, 'pretrained_weight')
    save_data(selected_bias, 'pretrained_bias')


def save_data(x, name):
    clone = torch.detach(x)
    torch.save(clone, 'export/' + name + '.pth')
    return x


def get_model_with_export_function(model):
    symbolic_module = fx.symbolic_trace(model)
    symbolic_module_dict = dict(symbolic_module.named_modules())
    symbolic_module_graph = symbolic_module.graph

    for current_node in symbolic_module_graph.nodes:
        if current_node.name == 'mid_part_8':
            previous_origin_node = current_node.prev
            with symbolic_module_graph.inserting_before(current_node):
                new_node = symbolic_module_graph.call_function(save_data, kwargs={'name': 'input'})
                set_node_input(new_node, get_node_output(previous_origin_node))
                new_node_output = get_node_output(new_node)
                set_node_input(current_node, new_node_output)

            next_origin_node = current_node.next
            with symbolic_module_graph.inserting_after(current_node):
                new_node = symbolic_module_graph.call_function(save_data, kwargs={'name': 'output'})
                set_node_input(new_node, get_node_output(current_node))
                new_node_output = get_node_output(new_node)
                set_node_input(next_origin_node, new_node_output)

            break

    symbolic_module_graph.lint()
    return torch.fx.GraphModule(model, symbolic_module_graph)


def export_data():
    model = FSRCNN(scale_factor=SCALE).eval()
    model.load_state_dict(torch.load('data/checkpoints/fsrcnn_x3.pth', map_location=device))
    model_with_export_function = get_model_with_export_function(model)

    image = Image.open('data/samples/butterfly_GT.bmp').convert('RGB')
    image_width = 288
    image_height = 288
    hr = image.resize((image_width, image_height), resample=Image.BICUBIC)
    lr = hr.resize((hr.width // SCALE, hr.height // SCALE), resample=Image.BICUBIC)
    input_data, _ = preprocess(lr, device)

    pred = model_with_export_function(input_data)


def quantize_tensor(tensor_input: torch.Tensor, width: int):
    assert isinstance(tensor_input, torch.Tensor)
    assert isinstance(width, int)
    assert width > 0

    max_val = torch.max(tensor_input).item()
    min_val = torch.min(tensor_input).item()

    tensor_element_range = max_val - min_val

    if tensor_element_range == 0:
        return tensor_input, 0, 0
    else:
        steps = 2 ** width
        resolution = tensor_element_range / steps

        tensor_steps = tensor_input / resolution
        tensor_steps_int = torch.floor(tensor_steps)

        quantized_tensor = tensor_steps_int * resolution

        return quantized_tensor, tensor_steps_int, resolution


def reshape_tensor_for_hardware_pe_input(input_tensor: torch.Tensor, pe_num: int = 4):
    input_dimension = len(input_tensor.shape)
    assert input_dimension == 4, 'Expect input tensor dimension: 4, but get %d' % input_dimension

    batch, channel, height, width = input_tensor.shape
    expanded_channel = channel
    if channel % pe_num != 0:
        expanded_channel = ((channel // pe_num) + 1) * pe_num

    tensor_buffer = torch.zeros([batch, expanded_channel, height, width])
    tensor_buffer[:, 0: channel, :, :] = input_tensor
    output_tensor = torch.zeros(batch * pe_num, channel, height, width)

    for current_batch in range(batch):
        for current_pe in range(pe_num):
            target_batch = current_batch * pe_num + current_pe
            target_non_zero_channel_num = (channel + pe_num - 1) // pe_num
            target_channel_start = current_pe * target_non_zero_channel_num
            target_channel_end = min((current_pe + 1) * target_non_zero_channel_num, channel)

            output_tensor[target_batch, target_channel_start: target_channel_end, :, :] \
                = tensor_buffer[current_batch, target_channel_start: target_channel_end, :, :]
    return output_tensor


def main():
    pretrained_weight = torch.load('export/pretrained_weight.pth')
    pretrained_bias = torch.load('export/pretrained_bias.pth')

    weight_quantized, weight_int, weight_scale = quantize_tensor(pretrained_weight, WEIGHT_WIDTH)
    bias_quantized, bias_int, bias_scale = quantize_tensor(pretrained_bias, BIAS_WIDTH)

    model = HardwareModel()
    model.conv1.weight.data = pretrained_weight
    # model.conv1.bias.data = pretrained_bias

    model_quantized = HardwareModel()
    model_quantized.conv1.weight.data = weight_quantized
    # model_quantized.conv1.bias.data = bias_quantized

    model_int = HardwareModel()
    model_int.conv1.weight.data = weight_int
    # model_quantized.conv1.bias.data = bias_quantized

    input_data = torch.load('export/input.pth')
    output_reference_data_from_fsrcnn = torch.load('export/output.pth')

    input_quantized, input_int, input_scale = quantize_tensor(input_data, INPUT_WIDTH)

    output_original = model(input_data)
    output_int = model_int(input_int)
    output_quantized = model_quantized(input_quantized)

    output_int = model_int(input_int)
    output_scale = weight_scale * input_scale
    output_quantized_from_int = output_int * output_scale

    input_patch_32_original_list = []
    input_patch_32_int_list = []
    input_patch_32_quantized_list = []
    for x in range(96 - 32 + 1):
        for y in range(96 - 32 + 1):
            input_patch_32_original_list.append(input_data[:, :, x: x + 32, y: y + 32])
            input_patch_32_int_list.append(input_int[:, :, x: x + 32, y: y + 32].int())
            input_patch_32_quantized_list.append(input_quantized[:, :, x: x + 32, y: y + 32])

    input_patch_original_list = []
    input_patch_int_list = []
    input_patch_quantized_list = []
    for index in range(len(input_patch_32_original_list)):
        input_patch_original_list.append(reshape_tensor_for_hardware_pe_input(input_patch_32_original_list[index]))
        input_patch_int_list.append(reshape_tensor_for_hardware_pe_input(input_patch_32_int_list[index]))
        input_patch_quantized_list.append(reshape_tensor_for_hardware_pe_input(input_patch_32_quantized_list[index]))

    output_patch_original_list = []
    output_patch_int_list = []
    output_patch_quantized_list = []
    for index in range(len(input_patch_original_list)):
        output_patch_original = model(input_patch_original_list[index])
        output_patch_int = model_int(input_patch_int_list[index]).int()
        output_patch_quantized = model_quantized(input_patch_quantized_list[index])

        output_patch_original_list.append({
            'reg_1': [output_patch_original[0], output_patch_original[1], output_patch_original[2],
                      output_patch_original[3]],
            'reg_2': [output_patch_original[0] + output_patch_original[1],
                      output_patch_original[2] + output_patch_original[3]],
            'reg_3': output_patch_original[0] + output_patch_original[1] + output_patch_original[2] +
                     output_patch_original[3]
        })
        output_patch_int_list.append({
            'reg_1': [output_patch_int[0], output_patch_int[1], output_patch_int[2], output_patch_int[3]],
            'reg_2': [output_patch_int[0] + output_patch_int[1], output_patch_int[2] + output_patch_int[3]],
            'reg_3': output_patch_int[0] + output_patch_int[1] + output_patch_int[2] + output_patch_int[3]
        })
        output_patch_quantized_list.append({
            'reg_1': [output_patch_quantized[0], output_patch_quantized[1], output_patch_quantized[2],
                      output_patch_quantized[3]],
            'reg_2': [output_patch_quantized[0] + output_patch_quantized[1],
                      output_patch_quantized[2] + output_patch_quantized[3]],
            'reg_3': output_patch_quantized[0] + output_patch_quantized[1] + output_patch_quantized[2] +
                     output_patch_quantized[3]
        })

    weight_parameter_save_dict = {
        'weight_original_min': torch.min(pretrained_weight).item(),
        'weight_original_max': torch.max(pretrained_weight).item(),
        'weight_int_min': torch.min(weight_int).int().item(), 'weight_int_max': torch.max(weight_int).int().item(),
        'weight_quantized_min': torch.min(weight_quantized).item(),
        'weight_quantized_max': torch.max(weight_quantized).item(),
        'weight_scale': weight_scale, 'weight_zero': 0,
        'weight_width': WEIGHT_WIDTH
    }
    bias_parameter_save_dict = {
        'bias_original_min': torch.min(pretrained_bias).item(), 'bias_original_max': torch.max(pretrained_bias).item(),
        'bias_int_min': torch.min(bias_int).int().item(), 'bias_int_max': torch.max(bias_int).int().item(),
        'bias_quantized_min': torch.min(bias_quantized).item(), 'bias_quantized_max': torch.max(bias_quantized).item(),
        'bias_scale': bias_scale, 'bias_zero': 0,
        'bias_width': BIAS_WIDTH
    }
    input_parameter_save_dict = {
        'input_original_min': torch.min(input_data).item(), 'input_original_max': torch.max(input_data).item(),
        'input_int_min': torch.min(input_int).int().item(), 'input_int_max': torch.max(input_int).int().item(),
        'input_quantized_min': torch.min(input_quantized).item(),
        'input_quantized_max': torch.max(input_quantized).item(),
        'input_scale': input_scale, 'input_zero': 0,
        'input_width': INPUT_WIDTH
    }
    output_parameter_save_dict = {
        'output_original_min': torch.min(output_original).item(),
        'output_original_max': torch.max(output_original).item(),
        'output_int_min': torch.min(output_int).int().item(), 'output_int_max': torch.max(output_int).int().item(),
        'output_quantized_min': torch.min(output_quantized).item(),
        'output_quantized_max': torch.max(output_quantized).item(),
        'output_scale': input_scale * weight_scale, 'output_zero': 0,
        'output_width': {'reg_1': 16, 'reg_2': 17, 'reg_3': 18}
    }

    weight_save_dict = {
        'weight_original': pretrained_weight, 'weight_int': weight_int.int(), 'weight_quantized': weight_quantized,
        'weight_parameter_dict': weight_parameter_save_dict
    }
    bias_save_dict = {
        'bias_original': pretrained_bias, 'bias_int': bias_int.int(), 'bias_quantized': bias_quantized,
        'bias_parameter_dict': bias_parameter_save_dict
    }
    input_save_dict = {
        'input_original': input_data, 'input_int': input_int.int(), 'input_quantized': input_quantized,
        'input_original_patch_list': input_patch_original_list, 'input_int_patch_list': input_patch_int_list,
        'input_quantized_patch_list': input_patch_quantized_list,
        'input_parameter_dict': input_parameter_save_dict
    }
    output_save_dict = {
        'output_original': output_original, 'output_int': output_int.int(), 'output_quantized': output_quantized,
        'output_patch_original_list': output_patch_original_list, 'output_patch_int_list': output_patch_int_list,
        'output_patch_quantized_list': output_patch_quantized_list,
        'output_parameter_dict': output_parameter_save_dict
    }

    final_save_dict = {
        'weight': weight_save_dict,
        'bias': bias_save_dict,
        'input': input_save_dict,
        'output': output_save_dict
    }

    torch.save(final_save_dict, 'export/everything.pth')


if __name__ == '__main__':
    export_data()
    main()

