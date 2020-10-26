from __future__ import print_function

import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from models import GC_MRNet
from configs import Config
from datasets import MyDatasets


def test(model, val_loader, use_cuda):
    if use_cuda:
        model = model.cuda()

    total1, total2 = [], []
    test_log = open(cfg.lof_dir + '/ST-A_test.txt', mode='w')
    model.eval()
    with torch.no_grad():
        for j, (img, target_dmp, target_cnt) in tqdm(enumerate(val_loader)):
            if use_cuda:
                img = img.cuda()
                target_dmp = target_dmp.cuda()
                target_cnt = target_cnt.cuda()

            dmp1, dmp2 = model(img)
            gt_cnt = target_cnt.sum()

            out_cnt1 = dmp1.sum()
            out_cnt2 = dmp2.sum()

            ae1 = torch.abs(out_cnt1 - gt_cnt)
            se1 = torch.pow(out_cnt1 - gt_cnt, 2)

            ae2 = torch.abs(out_cnt2 - gt_cnt)
            se2 = torch.pow(out_cnt2 - gt_cnt, 2)

            total1.append([ae1.item(), se1.item()])
            total2.append([ae2.item(), se2.item()])

            line = 'out1:%.3f, out2:%.3f,  gt:%.3f' % (out_cnt1.item(), out_cnt2.item(), gt_cnt.item())
            test_log.write(line + '\n')

        mae1 = np.asarray(total1)[:, 0].mean()
        mse1 = np.sqrt(np.asarray(total1)[:, 1].mean())
        mae2 = np.asarray(total2)[:, 0].mean()
        mse2 = np.sqrt(np.asarray(total2)[:, 1].mean())

        line = str('Stage-1: mae=%.3f, mse=%.3f, Stage-2: mae=%.3f, mse=%.3f' % (mae1, mse1, mae2, mse2))
        test_log.write(line)
        print(line)
        test_log.close()


if __name__ == '__main__':
    cfg = Config()
    use_cuda = torch.cuda.is_available()
    val_dataset = MyDatasets(cfg=cfg, group='validation')
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)
    model = GC_MRNet(cfg.best_ckpt)
    test(model, val_loader, use_cuda)