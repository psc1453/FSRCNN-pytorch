import torch
from torch.fx import symbolic_trace
from torch.fx.node import Node

from models import FSRCNN
from quant_utils import quantize_model_parameters_with_original_scale, quantize_tensor_with_original_scale,NN


CKPT_PATH = 'data/checkpoints/fsrcnn_x3.pth'
SCALE = 3


def get_node_input(node: Node):
    return node.args


def set_node_input(node: Node, value):
    node.args = (value,)


def get_node_output(node: Node) -> Node:
    return node


def generate_quantized_module(model_input: NN, insert_function, parameter_dict) -> torch.fx.GraphModule:
    # Generate necessary components
    symbolic_traced_module = symbolic_trace(model_input)
    symbolic_traced_module_dict = dict(symbolic_traced_module.named_modules())
    symbolic_traced_module_graph = symbolic_traced_module.graph

    insert_pattern_list = [torch.nn.Conv2d]

    def node_match(node_input, pattern_list_input):
        for pattern in insert_pattern_list:
            if node_input.target in symbolic_traced_module_dict:
                if type(symbolic_traced_module_dict[node_input.target]) is pattern:
                    return True

    last_node = None
    last_origin_node_has_been_inserted = False
    latest_node_is_new_inserted = False
    for current_node in symbolic_traced_module_graph.nodes:
        # Skip an iteration if the last iteration inserts a new node, because this iteration is the new node
        if latest_node_is_new_inserted:
            # Next iteration will not enter this branch, it will be the originally existed node
            latest_node_is_new_inserted = False
        # Only originally existed node can Enter this branch.
        else:
            if last_origin_node_has_been_inserted:
                set_node_input(current_node, get_node_output(last_node))
            else:
                last_origin_node_has_been_inserted = True
            # If this node match the patter, a new node needs to be inserted after it
            if node_match(current_node, [insert_pattern_list]):
                # Create temporary pointer for inserting
                with symbolic_traced_module_graph.inserting_after(current_node):
                    # Create new node after current node
                    new_node = symbolic_traced_module_graph.call_function(insert_function)
                    # Set the input of the new node to the output of the current node
                    set_node_input(new_node, get_node_output(current_node))
                    # Pass parameters to the inserted function
                    new_node.kwargs = parameter_dict
                    # Update pointer
                    last_node = new_node
                    # Latest node becomes the newly inserted one, and will belong to next iteration
                    # Should skip that iteration
                    latest_node_is_new_inserted = True
            # Doesn't match the pattern,
            else:
                # Just update the pointer
                last_node = current_node
                # Latest node it the current one, and will belong to next iteration
                # Shouldn't skip that iteration
                latest_node_is_new_inserted = False

    symbolic_traced_module_graph.lint()
    return torch.fx.GraphModule(symbolic_traced_module, symbolic_traced_module_graph)


if __name__ == '__main__':
    test_input = torch.randn([1, 1, 100, 100])

    model = FSRCNN(scale_factor=SCALE)
    quantized_by_parameters_model = quantize_model_parameters_with_original_scale(model_input=model, weight_width=8,
                                                                                  bias_weight=18)

    new = generate_quantized_module(model_input=quantized_by_parameters_model,
                                    insert_function=quantize_tensor_with_original_scale, parameter_dict={'width': 8})

    new.print_readable()
    print(model(test_input))
    print(new(test_input))
