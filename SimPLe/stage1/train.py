import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
import sys
import shutil
from data import BreastTumor, data_prefetcher
from data_transform import Norm, RandomCrop, ToTensor

sys.path.append('..')
from net import Unet
from file_and_folder_operations import myMakedirs

def reproduce(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)

def main():
    reproduce(args.seed)
    logging.basicConfig(filename=os.path.join(args.exp_name, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = Unet(1, 2).cuda()

    train_data_list = os.listdir('/data/zym/workspace/noisy/noisy_label_resample_1.0X0.6X0.6')

    transform_fg_train = transforms.Compose([Norm(),
                                             RandomCrop(args.patch_size, 1, 1.),
                                             ToTensor(0)])
    train_fg_dataset = BreastTumor(train_data_list, transform=transform_fg_train)
    fg_dataloader = DataLoader(train_fg_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               pin_memory=False,
                               worker_init_fn=worker_init_fn,
                               drop_last=True)

    transform_bg_train = transforms.Compose([Norm(),
                                             RandomCrop(args.patch_size, 0, 1.),
                                             ToTensor(0)])
    train_bg_dataset = BreastTumor(train_data_list, transform=transform_bg_train)
    bg_dataloader = DataLoader(train_bg_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               pin_memory=False,
                               worker_init_fn=worker_init_fn,
                               drop_last=True)

    optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.99, weight_decay=1e-4, nesterov=True)
    writer = SummaryWriter(os.path.join(args.exp_name, 'tbx'))

    CE = torch.nn.CrossEntropyLoss(ignore_index=2)
    BCE = torch.nn.BCELoss()
    sigmoid = torch.nn.Sigmoid()

    iter_num = 0
    max_epoch = int(args.max_epoch)
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        epoch_num = epoch_num + 1

        fg_prefetcher = data_prefetcher(fg_dataloader)
        bg_prefetcher = data_prefetcher(bg_dataloader)
        fg_sample = fg_prefetcher.next()
        bg_sample = bg_prefetcher.next()
        while fg_sample is not None and bg_sample is not None:
            iter_num = iter_num + 1
            net.train()

            fg_img, fg_seg, fg_gt = fg_sample['image'], fg_sample['label'], fg_sample['gt']
            bg_img, bg_seg, bg_gt = bg_sample['image'], bg_sample['label'], bg_sample['gt']
            fg_img, fg_seg = fg_img.cuda(), fg_seg.cuda()
            bg_img, bg_seg = bg_img.cuda(), bg_seg.cuda()

            fg_outs, fg_fea = net(fg_img)
            bg_outs, _ = net(bg_img)

            fea = fg_fea.permute(0, 2, 3, 4, 1).contiguous()

            positive_mask = torch.zeros_like(fg_seg, dtype=torch.bool)
            positive_mask[fg_seg == 1] = 1
            positive_samples = fea[positive_mask]
            positive_samples = positive_samples[torch.randperm(positive_samples.size(0))]
            positive_samples = positive_samples[:100]

            negative_mask = torch.zeros_like(fg_seg, dtype=torch.bool)
            negative_mask[fg_seg == 0] = 1
            negative_samples = fea[negative_mask]
            negative_samples = negative_samples[torch.randperm(negative_samples.size(0))]
            negative_samples = negative_samples[:100]

            all_samples = torch.cat([positive_samples, negative_samples], dim=0)

            temperature = 0.1
            sim_m = F.cosine_similarity(all_samples[..., None, :, :], all_samples[..., :, None, :],
                                        dim=-1) / temperature
            sim_m = sigmoid(sim_m)

            mask = torch.triu(torch.ones(sim_m.shape[0], sim_m.shape[1]), diagonal=1)
            mask[100:200, 100:200] = 0

            label = torch.triu(torch.ones(sim_m.shape[0], sim_m.shape[1]), diagonal=1)
            label[0:100, 100:200] = 0

            vaild_sim_m = sim_m[mask == 1]
            vaild_label = label[mask == 1]
            contrastive_loss = BCE(vaild_sim_m, vaild_label.to(vaild_sim_m.device))

            loss = CE(fg_outs, fg_seg) + CE(bg_outs, bg_seg) + contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/L_supervised', loss.item(), iter_num)
            writer.add_scalar('loss/L_contrastive', contrastive_loss.item(), iter_num)

            if iter_num % 50 == 0:
                image = fg_img[0, 0:1, 10:51:10, :, :].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                outputs_soft = F.softmax(fg_outs, 1)
                image = outputs_soft[0, 1:2, 10:51:10, :, :].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted', grid_image, iter_num)

                gt_batch = fg_gt.long()
                image = gt_batch[0, 10:51:10, :, :].unsqueeze(0).permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth', grid_image, iter_num)

            fg_sample = fg_prefetcher.next()
            bg_sample = bg_prefetcher.next()

        lr_ = args.base_lr * (1 - epoch_num / max_epoch) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        # save
        if epoch_num % args.save_per_epoch == 0:
            save_model_path = os.path.join(args.exp_name, f'epoch_{epoch_num}.pth')
            torch.save(net.state_dict(), save_model_path)


    writer.close()

    save_model_path = os.path.join(args.exp_name, f'epoch_{max_epoch}.pth')
    torch.save(net.state_dict(), save_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='/data/zym/experiment/noisy/stage1')
    # parser.add_argument('--exp_name', type=str, default='/data/zym/experiment/noisy/DEBUG')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patch_size', type=list, default=[96, 128, 128])
    parser.add_argument('--base_lr', type=float, default=1e-2)
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--save_per_epoch', type=int, default=20)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.exp_name == '/data/zym/experiment/noisy/DEBUG':
        myMakedirs(args.exp_name, overwrite=True)
    else:
        myMakedirs(args.exp_name, overwrite=False)

    # save code
    py_path_old = os.path.dirname(os.path.abspath(sys.argv[0]))
    py_path_new = os.path.join(args.exp_name, 'code')
    shutil.copytree(py_path_old, py_path_new)

    main()
