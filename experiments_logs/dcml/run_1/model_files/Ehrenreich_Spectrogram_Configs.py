class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 80  # Mel spectrogram bands
        self.kernel_size = 5  # Increased from 8 for more aggressive downsampling
        self.stride = 2        # Increased from 1 for aggressive stride
        self.final_out_channels = 256  # Increased from 128

        self.num_classes = 2
        self.class_weights = [1.0, 10.0]
        self.dropout = 0.35
        self.features_len = 72  # for 6s windows
        # self.features_len = 49  # for 4s windows

        # training configs
        self.num_epoch = 40

        
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 64  # Reduced for spectrograms which are memory intensive

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 25
