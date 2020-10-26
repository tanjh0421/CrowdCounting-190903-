import os
import math
import random
import logging
import cv2 as cv
import numpy as np

from tqdm import tqdm
from time import time
from scipy.io import loadmat

from torch.utils.data import Dataset

from configs import Config
from tools import Tool

logging.basicConfig(level='INFO')


class MyDatasets(Dataset):
    def __init__(self, cfg, group, num_workers=4):
        self.cfg = cfg  # configurations #
        self.ul = Tool()  # some tools
        self.group = group  # train or validation #
        self.scale = 8
        self.max_size = 1920  # max size of image #
        self.items = self.get_datasets()
        self.num = len(self.items)  # number of samples #
        self.data = self.get_data()

        self.batch_size = cfg.batch_size
        self.num_workers = num_workers
        self.use_crop = cfg.use_crop
        self.use_flip = cfg.use_flip
        self.use_noise = cfg.use_noise
        self.use_color2gray = cfg.use_color2gray
        self.use_contrast = cfg.use_contrast

    def get_datasets(self):
        items = []
        if self.cfg.dataset_name == 'ST-A':
            if self.group == 'train':
                file = {'img_path': '/home/lhl/data/ShanghaiTech/part_A_final/train_data/images',
                        'mat_path': '/home/lhl/data/ShanghaiTech/part_A_final/train_data/ground_truth',
                        'mode': self.cfg.sta_mode,
                        }
            else:
                file = {'img_path': '/home/lhl/data/ShanghaiTech/part_A_final/test_data/images',
                        'mat_path': '/home/lhl/data/ShanghaiTech/part_A_final/test_data/ground_truth',
                        'mode': self.cfg.sta_mode,

                        }
        elif self.cfg.dataset_name == 'ST-B':
            if self.group == 'train':
                file = {'img_path': '/home/lhl/data/ShanghaiTech/part_B_final/train_data/images',
                        'mat_path': '/home/lhl/data/ShanghaiTech/part_B_final/train_data/ground_truth',
                        'mode': self.cfg.stb_mode,
                        }
            else:
                file = {'img_path': '/home/lhl/data/ShanghaiTech/part_B_final/test_data/images',
                        'mat_path': '/home/lhl/data/ShanghaiTech/part_B_final/test_data/ground_truth',
                        'mode': self.cfg.stb_mode,

                        }
        else:
            AssertionError('No datasets')
            file = {}

        name = os.listdir(file['img_path'])
        img_list = [os.path.join(file['img_path'], s) for s in name]
        mat_list = [os.path.join(file['mat_path'], 'GT_' + s.replace('jpg', 'mat')) for s in name]
        for i in range(len(img_list)):
            img = cv.imread(img_list[i])
            pos = loadmat(mat_list[i])['image_info'][0][0]['location'][0][0]
            cnt = pos.shape[0]
            dt = {'img': img, 'pos': pos, 'cnt': cnt, 'mode': file['mode']}
            items.append(dt)
        return items

    def get_data(self):
        logging.info('Making density map, (%s) samples total, ......' % str(self.num))
        end = time()
        data = []
        for i, item in tqdm(enumerate(self.items)):
            # get images,position, ground truth data
            img, pos, gt_cnt = item['img'], item['pos'], item['cnt']
            # RGB format
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # for saving memory using resize to limit the image size
            scale = 1
            h, w = img.shape[0], img.shape[1]
            if h > self.max_size or w > self.max_size:
                scale = (h / self.max_size) if h > w else (w / self.max_size)
                h, w = math.ceil(h / scale), math.ceil(w / scale)
                img = cv.resize(img, (w, h))
                pos = pos / scale

            # only the region of interest is concerned
            if 'roi' in item:
                img_roi = np.zeros(img.shape[:2])

                pos = self.ul.points_in_polygon((item['roi'] / scale).astype('int'), pos)

                img_roi = cv.fillConvexPoly(img_roi, pos.astype('int'), 1)
                img_roi = img_roi[:, :, None].repeat(3, axis=2)
                img = (img * img_roi).astype('uint8')

                gt_cnt = pos.shape[0]

            # get the corresponding density map as same size as image
            h, w = img.shape[0], img.shape[1]
            if self.group == 'train':
                gt_dmp = self.ul.get_density_map((h, w), pos, item['mode'], self.cfg.knn, self.cfg.lim)
                gt_dmp = np.expand_dims(gt_dmp, axis=0).astype(np.float32)
            else:
                # ToDO:: make density map of test set (to save memory, no to make density map during test phrase now)
                gt_dmp = np.ones((1, 1))

            img = np.transpose(img, axes=[2, 0, 1]).astype(np.float32)
            gt_cnt = np.reshape(np.asarray(gt_cnt), (1,)).astype(np.float32)
            data.append(dict(img=img, gt_dmp=gt_dmp, gt_cnt=gt_cnt))

        end = time() - end
        logging.info('Finished, cost time %.3f total.' % end)

        return data

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img, target_dmp, target_cnt = self.data[index]['img'], self.data[index]['gt_dmp'], self.data[index]['gt_cnt']
        # if self.transform is not None:

        if self.group == 'train':
            # 9 times crop
            if self.use_crop:
                img, target_dmp, target_cnt = self.__crop_aug(img, target_dmp)

            h, w = img.shape[1], img.shape[2]
            target_dmp = cv.resize(target_dmp[0], (math.floor(w / self.scale), math.floor(h / self.scale)),
                                   interpolation=cv.INTER_CUBIC) * self.scale ** 2
            target_dmp = np.expand_dims(target_dmp, axis=0)

            if self.use_flip:
                img, target_dmp = self.__flip_aug(img, target_dmp)

            if self.use_noise:
                img = self.__noise_aug(img)

            if self.use_color2gray:
                img = self.__color2gray_aug(img)

            if self.use_contrast:
                img = self.__contrast_aug(img)

        else:
            # ToDo:: reszie density map of test set (to save memory, no to make density map during test phrase now)
            h, w = img.shape[1], img.shape[2]
            # target_dmp = cv.resize(target_dmp[0], (math.floor(w / self.scale), math.floor(h / self.scale)),
            #                        interpolation=cv.INTER_CUBIC) * self.scale ** 2
            # target_dmp = np.expand_dims(target_dmp, axis=0)

        return img, target_dmp, target_cnt

    def __crop_aug(self, img, target_dmp):
        h, w = img.shape[1], img.shape[2]

        lt_corner = [(0, 0), (h // 2, 0), (0, w // 2), (h // 2, w // 2),
                     (random.randint(0, h // 2), random.randint(0, w // 2)),
                     (random.randint(0, h // 2), random.randint(0, w // 2)),
                     (random.randint(0, h // 2), random.randint(0, w // 2)),
                     (random.randint(0, h // 2), random.randint(0, w // 2)),
                     (random.randint(0, h // 2), random.randint(0, w // 2))]
        index = random.randint(0, len(lt_corner) - 1)
        y0, y1 = lt_corner[index][0], lt_corner[index][0] + h // 2
        x0, x1 = lt_corner[index][1], lt_corner[index][1] + w // 2
        img = img[:, y0:y1, x0:x1]
        target_dmp = target_dmp[0, y0:y1, x0:x1]
        target_cnt = np.reshape(np.sum(target_dmp, axis=(0, 1)), newshape=(-1, 1))
        target_dmp = np.expand_dims(target_dmp, axis=0)
        return img, target_dmp, target_cnt

    def __flip_aug(self, img, target_dmp):
        if random.random() > 0.8:
            img = np.fliplr(img).copy()
            target_dmp = np.fliplr(target_dmp).copy()
        return img, target_dmp

    def __noise_aug(self, img):
        if random.random() > 0.7:
            if random.random() > 0.5:
                img = np.power(img / float(np.max(img)), 0.5)
            else:
                img = np.power(img / float(np.max(img)), 1.5)
        return img

    def __contrast_aug(self, img):
        if random.random() > 0.5:
            if random.random() < 0.5:
                img = np.float32(np.uint8(np.clip((1.2 * img + 10), 0, 255)))
            elif random.random() < 0.7:
                img = np.float32(np.uint8(np.clip((1.5 * img + 10), 0, 255)))
            else:
                img = np.float32(np.uint8(np.clip((1.8 * img + 10), 0, 255)))
        return img

    def __color2gray_aug(self, img):
        if random.random() > 0.9:
            img = cv.cvtColor(np.transpose(img, axes=(1, 2, 0)), cv.COLOR_RGB2GRAY)
            img = np.tile(np.expand_dims(img, axis=2), 3)
            img = np.transpose(img, axes=(2, 0, 1))
        return img


if __name__ == '__main__':
    cfg = Config()
    # validation
    dataset = MyDatasets(cfg, 'validation')
    # dataset = MyDatasets(cfg, 'train')
    img, dmp, cnt = dataset.__getitem__(1)
    print(img.shape, dmp.shape, cnt.shape, dmp.sum(), cnt)
    print(dataset.num)
