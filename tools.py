import numpy as np
import math
from matplotlib.path import Path

np.set_printoptions(threshold=np.inf)


class Tool(object):
    def __init__(self):
        self.crop_size = 256

    def fspecial(self, rows, cols, sigma):
        """
        2D-Gaussian kernel
        :param rows: the height of 2D-Gaussian kernel size int(kernel_row)
        :param cols: the weight of 2D-Gaussian kernel size int(kernel_col)
        :param sigma: the spread parameter (standard variance) of 2D-Gaussian kernel float(sigma)
        :return: normalized 2D-Gaussian kernel (height, wight)
        """
        x, y = np.mgrid[-rows / 2 + 0.5: rows // 2 + 0.5, -cols / 2 + 0.5: cols // 2 + 0.5]
        gaussian_kernel = np.exp(-(np.square(x) + np.square(y)) / (2 * np.power(sigma, 2))) / (2 * np.power(sigma, 2))
        norm = gaussian_kernel / gaussian_kernel.sum()

        return norm

    def knn(self, x, y, points, k):
        """
        Euclidean Metric of mean k nearest neightbor
        :param x: a specific point position x float(x)
        :param y: a specific point position y float(y)
        :param points: the labeled points for the head center of each pedestrian on crowd image its dimension (number_of_points, 2)
        :param k: the k nearest neighbors int(k)
        :return: Euclidean Metric of mean k nearest neightbor float(mean_k_dice)
        """
        num = len(points)
        if k >= num >= 2:
            k = num - 1
        elif num < 2:
            return 1.0
        dice = np.zeros((num, 1), dtype=np.float)
        for i in range(num):
            x1 = points[i, 0]
            y1 = points[i, 1]
            # compute Euclidean Metric of each points with point(x, y)
            dice[i, 0] = math.sqrt(math.pow(x - x1, 2) + math.pow(y - y1, 2))
        # sort by descending order
        dice[:, 0] = np.sort(dice[:, 0])
        # compute the mean k nearest neighbor distance
        mean_k_dice = np.mean(dice[1:k + 1, 0])

        return mean_k_dice

    def get_density_map(self, dmp_size, points, gaussian_mode, k, lim):
        """
        generate ground truth density_map for training
        :param dmp_size: list or tuple, the size of density map (height, width) or [height, width]
        :param points: np.ndarray, the labeled points for the head center of each pedestrian on crowd image its dimension (number_of_points, 2)
        :param gaussian_mode: dict, define whether is is geometry-adaptive gaussian kernel or not, dict(mode='constant', sigma=25, beta=1.0)
        :param k: int, the k nearest neighbors int(k)
        :param lim: dict, the [min, max] limitation of pedestrian head size
                    for estimated head size by k nearest neighbor, dict(min_lim=1.0, max_lim=100.0)
        :return: np.ndarray, density_map (height, width)
        """
        # the size of density map
        h, w = dmp_size[0], dmp_size[1]
        # truncate = 4.0
        density_map = np.zeros((h, w))

        num = len(points)
        if num == 0:
            return density_map

        for i in range(num):
            x = min(w, max(0, abs(int(math.floor(points[i, 0])))))
            y = min(h, max(0, abs(int(math.floor(points[i, 1])))))
            # if fixed, fixed the 2D-Gaussian kernel;
            # else use the geometry-adaptive Gaussian kernel
            if gaussian_mode['mode'] == 'constant':
                sigma = gaussian_mode['sigma']
            elif gaussian_mode == 'adaptive':
                km_dist = max(lim['min_lim'], min(self.knn(x, y, points, k), lim['max_lim']))
                sigma = gaussian_mode['beta'] * km_dist
            else:
                sigma = 15

            # ksize = 2 * int(truncate * sigma + 0.5)
            ksize = 15

            radius = ksize / 2
            x1 = x - int(math.floor(radius))
            y1 = y - int(math.floor(radius))
            x2 = x + int(math.ceil(radius))
            y2 = y + int(math.ceil(radius))

            # address edge
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)

            kcol = x2 - x1
            krow = y2 - y1

            gaussian_kernel_2d = self.fspecial(krow, kcol, sigma)
            density_map[y1:y2, x1:x2] = density_map[y1:y2, x1:x2] + gaussian_kernel_2d

        return density_map

    def points_in_polygon(self, verts, points):
        ph = Path(verts, closed=True)
        is_contain = ph.contains_points(points, radius=-0.05)
        points_index = np.argwhere(is_contain == True)
        points_contain = points[points_index]
        return points_contain[:, 0, :]

