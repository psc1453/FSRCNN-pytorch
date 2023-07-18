import argparse

import torch
import numpy as np
import PIL.Image as pil_image

from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr

from ModelModifier.modifier.classes import NodeInsertMapping, NodeInsertMappingElement, FunctionPackage
from ModelModifier.modifier.utils import generate_quantized_module
from ModelModifier.tools.quantization import quantize_model_parameters_with_original_scale, \
    quantize_tensor_with_original_scale

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='data/checkpoints/fsrcnn_x4.pth')
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    device = 'cpu'

    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for layer_name, layer_parameter in torch.load(args.weights_file, map_location=device).items():
        if layer_name in state_dict.keys():
            state_dict[layer_name].copy_(layer_parameter)
        else:
            raise KeyError(layer_name)

    model.eval()

    quantized_by_parameters_model = quantize_model_parameters_with_original_scale(model_input=model, weight_width=8,
                                                                                  bias_width=8)
    mapping = NodeInsertMapping()
    quantize_8bit_function_package = FunctionPackage(quantize_tensor_with_original_scale, {'width': 8})
    conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, quantize_8bit_function_package)
    mapping.add_config(conv2d_config)

    new = generate_quantized_module(model_input=quantized_by_parameters_model, insert_mapping=mapping)

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = new(lr).clamp(0.0, 1.0)

    psnr = calc_psnr(hr, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_fsrcnn_x{}.'.format(args.scale)))
