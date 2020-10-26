import os


class Config:
    """
    crowd counting configuration
    include: training hyper-parameters;
             Gaussian kernel parameters;
             log path;
             model path
    """
    def __init__(self):

        # about model configurations setting
        self.lr = 5e-6
        self.weight_decay = 5e-4
        self.epochs = 4000
        self.batch_size = 1
        self.input_channel = 3
        self.output_channel = 1

        # validation configurations setting
        self.snap = 5
        self.pretrained = True

        # training configurations setting
        self.use_crop = True
        self.use_flip = True
        self.use_noise = True
        self.use_color2gray = True
        self.use_contrast = True

        # Gaussian kernel parameters setting
        self.knn = 3
        self.lim = dict(min_lim=2.0, max_lim=100.0)

        self.sta_mode = dict(mode='constant', sigma=4.0, beta=0.3)
        self.stb_mode = dict(mode='constant', sigma=15.0, beta=1.0)

        # log and model saving path
        self.model_name = 'GC-MRNet'
        self.dataset_name = 'ST-A'
        self.log_dir = os.path.join(self.dataset_name, 'train_log/logs')
        self.model_dir = os.path.join(self.dataset_name, 'train_log/models')
        self.summary_dir = os.path.join(self.dataset_name, 'train_log/events')

        # summary
        self.train_log = os.path.join(self.log_dir, 'train.txt')
        self.val_log = os.path.join(self.log_dir, 'val.txt')

        # checkpoints
        self.latest_ckpt = self.model_dir + '/latest.pkl'
        self.best_ckpt = self.model_dir + '/best.pkl'

        self.__make_dir(self.log_dir)
        self.__make_dir(self.model_dir)
        self.__make_dir(self.summary_dir)

    def __make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

