from __future__ import print_function

import os
import random
import logging
import argparse
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
torch.cuda.set_device(7)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed = 0
set_seed(seed)

from models import Effcient
from configs import Config
from datasets import MyDatasets


def main(cfg, args):
    BEST_MAE = args.best_mae
    model = Effcient(pr=None)

    # dataset
    train_dataset = MyDatasets(cfg=cfg, group='train')
    val_dataset = MyDatasets(cfg=cfg, group='validation')
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size)

    criterion = nn.MSELoss(reduction='sum')
    criterion2 = nn.L1Loss(reduction='sum')

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion2 = criterion2.cuda()

    optimizer = torch.optim.Adam(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)

    val_log = open(cfg.val_log, mode='a')
    # reload model parameters
    if os.path.exists(cfg.latest_ckpt):
        logging.info('Load latest model...')
        model_dict = model.state_dict()
        model_v = torch.load(cfg.latest_ckpt)
        model_dict.update(model_v.state_dict())
        model.load_state_dict(model_dict)

    # Training phase
    for epoch in tqdm(range(1, cfg.epochs + 1)):
        model.train()
        tloss = []
        for i, (img, target_dmp, target_cnt) in tqdm(enumerate(train_loader)):
            if use_cuda:
                img = img.cuda()
                target_dmp = target_dmp.cuda()
                target_cnt = target_cnt.cuda()

            dmp = model(img)

            loss = criterion(dmp, target_dmp) + lc_loss(criterion2, dmp, target_dmp)
            tloss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [%d / %d], tloss=%.3f' % (epoch, cfg.epochs, np.asarray(tloss).mean()))

        # Validation phase
        if epoch % cfg.snap == 0 and epoch != 0:
            torch.save(model, cfg.latest_ckpt)
            model.eval()

            with torch.no_grad():
                total = []
                for j, (img, target_dmp, target_cnt) in tqdm(enumerate(val_loader)):
                    if use_cuda:
                        img = img.cuda()
                        target_dmp = target_dmp.cuda()
                        target_cnt = target_cnt.cuda()

                    dmp = model(img)
                    output = dmp
                    ae = torch.abs(torch.sum(output) - torch.sum(target_cnt))
                    se = torch.pow(torch.sum(output) - torch.sum(target_cnt), 2)
                    total.append([ae.cpu().detach().numpy(), se.cpu().detach().numpy()])

                mae = np.asarray(total)[:, 0].mean()
                mse = np.sqrt(np.asarray(total)[:, 1].mean())

                if mae < BEST_MAE:
                    BEST_MAE = mae
                    torch.save(model, cfg.best_ckpt)

                line = str('Epoch %d, mae=%.3f, mse=%.3f, best mae is %.3f' % (epoch, mae, mse, BEST_MAE))
                val_log.write(line + '\n')
                val_log.flush()
                print(line)

    val_log.close()


def lc_loss(cri, target_dmp, est_dmp):
    lc = cri(F.adaptive_avg_pool2d(est_dmp, (1, 1)), F.adaptive_avg_pool2d(target_dmp, (1, 1))) / 1.0 + \
               cri(F.adaptive_avg_pool2d(est_dmp, (2, 2)), F.adaptive_avg_pool2d(target_dmp, (2, 2))) / 4.0 + \
               cri(F.adaptive_avg_pool2d(est_dmp, (4, 4)), F.adaptive_avg_pool2d(target_dmp, (4, 4))) / 16.0
    return 1000 * lc / 3.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--best_mae', default=1E7, type=float, help='the best performance.')
    cfg = Config()
    args = parser.parse_args()
    main(cfg, args)
