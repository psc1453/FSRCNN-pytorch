import argparse

import torch
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import FSRCNN
from datasets import EvalDataset
from utils import AverageMeter, calc_psnr

from ModelModifier.modifier.classes import NodeInsertMapping, NodeInsertMappingElement, FunctionPackage
from ModelModifier.modifier.utils import insert_before
from ModelModifier.tools.hardware.bypass_bias import insert_bias_bypass, bypass_bias_adder
from ModelModifier.tools.quantization.utils import quantize_model_parameters_with_original_scale
from ModelModifier.tools.hardware.io import reshape_tensor_for_hardware_pe_input


def eval_one_epoch(model, dataloader, device, args):
    epoch_psnr = AverageMeter()

    for data in dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        if args.show:
            image_array = preds[0][0].cpu().numpy()
            plt.imshow(image_array, cmap='gray')
            plt.axis('off')
            plt.show()
        current_psnr = calc_psnr(preds, labels)
        epoch_psnr.update(current_psnr, len(inputs))
    return epoch_psnr.avg


def get_quant_model(model, weight=8, bias=18, conv=32, width=32):
    quantized_by_parameters_model = quantize_model_parameters_with_original_scale(model_input=model,
                                                                                  weight_width=weight,
                                                                                  bias_width=bias)
    before_mapping = NodeInsertMapping()
    reshape_function_package = FunctionPackage(reshape_tensor_for_hardware_pe_input)
    conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
    before_mapping.add_config(conv2d_config)

    bypass_mapping = NodeInsertMapping()
    reshape_function_package = FunctionPackage(bypass_bias_adder, {'width': width})
    conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
    bypass_mapping.add_config(conv2d_config)

    new = insert_before(model_input=quantized_by_parameters_model, insert_mapping=before_mapping)
    # model = insert_after(model_input=model, insert_mapping=after_mapping)
    new = insert_bias_bypass(model_input=new, insert_mapping=bypass_mapping)
    return new


def main(args):
    device = torch.device('cpu')

    model = FSRCNN(scale_factor=args.scale).to(device)
    model.load_state_dict(torch.load(args.weights_file, map_location=device))

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    model.eval()

    # fig, ax = plt.subplots(2, 1)
    plt.figure(figsize=(10, 6))
    plt.suptitle('Width of Accumulator vs PSNR (FSRCNN)', fontsize=16)

    psnr_list = {}
    for width in tqdm(range(1, 32)):
        model_quant = get_quant_model(model, width=width)
        psnr = eval_one_epoch(model_quant, eval_dataloader, device, args)
        psnr_list.update({width: psnr.cpu()})

    plt.subplot(1, 2, 1)
    plt.plot(psnr_list.keys(), psnr_list.values())
    plt.xlabel('Conv Width/Bits')
    plt.ylabel('PSNR/dB')
    plt.grid(alpha=0.4, linestyle=':')

    valuable_psnr_list = {k: v for k, v in psnr_list.items() if k in range(3, 16)}
    plt.subplot(1, 2, 2)
    plt.plot(valuable_psnr_list.keys(), valuable_psnr_list.values())
    plt.xlabel('Accumulator Width/Bits')
    plt.ylabel('PSNR/dB')
    plt.grid(alpha=0.4, linestyle=':')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', type=str, default='data/datasets/eval/Set5_x3.h5')
    parser.add_argument('--weights-file', type=str, default='data/checkpoints/fsrcnn_x3.pth')
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    main(args)
