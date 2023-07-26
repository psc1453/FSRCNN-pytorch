import torch
import argparse
from models import FSRCNN
import matplotlib.pyplot as plt


def main(args):

    model = FSRCNN(scale_factor=args.scale).eval()
    device = 'cpu'
    model.load_state_dict(torch.load(args.weights_file, map_location=device))
    # TODO: named_parameters() comments example wrong
    parameter_dict = model.named_parameters()
    for name, parameter in parameter_dict:
        if 'weight' in name and len(parameter.shape) == 4:
            print(name, ': ', parameter.shape)
            min_val = torch.min(parameter).item()
            max_val = torch.max(parameter).item()
            channel_num = parameter.shape[0]
            kernel_list = []

            plt.figure(figsize=(8, 120))
            for kernel_index in range(channel_num):
                kernel = parameter[kernel_index].detach().numpy().flatten()
                kernel_list.append(kernel)
                plt.subplot(channel_num, 1, kernel_index + 1)
                plt.hist(kernel, range=(min_val, max_val), bins=20)

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