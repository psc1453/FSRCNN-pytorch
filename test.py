import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

from models import FSRCNN
from datasets import EvalDataset
from utils import AverageMeter, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3, required=True)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()


    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN(scale_factor=args.scale).to(device)
    model.load_state_dict(torch.load(args.weights_file, map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, num_workers=args.num_workers)

    model.eval()

    epoch_psnr = AverageMeter()

    for data in eval_dataloader:
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

    print('Average Test PSNR: {:.2f}'.format(epoch_psnr.avg))
