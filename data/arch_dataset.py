import os
import torch
from util.condition import Condition
from data.aligned_dataset import AlignedDataset

class ArchDataset(AlignedDataset):
    def initialize(self, opt):
        self.opt = opt
        self.condition_history = Condition(opt)
        self.folders_label = opt.folders_label.split(';')
        self.folders_image = opt.folders_image.split(';')
        assert len(self.folders_image) == len(self.folders_label)

        ### input A (label maps)
        self.A_paths = []
        ### input B (real images)
        self.B_paths = []
        for folder_label, folder_image in zip(self.folders_label, self.folders_image):
            # 选择少的一个文件夹作为查找目录
            image_files = set(os.listdir(folder_image))
            label_files = set(os.listdir(folder_label))
            files = image_files & label_files
            for file in files:
                self.A_paths.append(os.path.join(folder_label, file))
                self.B_paths.append(os.path.join(folder_image, file))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        input_dict = super().__getitem__(index)
        # 添加条件
        condition = self.condition_history.get(self.B_paths[index])
        input_dict['condition'] = torch.tensor(condition, dtype=torch.float32)
        return input_dict