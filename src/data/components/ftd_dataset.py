import torch
from torch.utils.data import Dataset

class FtdDataset(Dataset):
    def __init__(
        self,
        input_sequences,  # 序列数据 (num_samples, seq_len, features)
        output_targets,    # 目标值 (num_samples, output_size)
        is_train,
        epoch_per_size,
        train_number=100000
    ):
        self.input_sequences = input_sequences
        self.output_targets = output_targets
        self.is_train = is_train
        self.epoch_per_size = epoch_per_size
        self.train_number = train_number
        self.num_per_epoch=int(len(self.input_sequences)/train_number)

    def __getitem__(self, index):
        if self.is_train:
            # 训练时随机选择样本
            permuted_indices = torch.randperm(len(self.input_sequences))
            permuted_indices = permuted_indices[:self.train_number]
            return self.input_sequences[permuted_indices], self.output_targets[permuted_indices]
        # 验证和测试时返回整个数据集
        else:
            initial_index=index*self.train_number
            final_index=(index+1)*self.train_number
            permuted_indices = torch.arange(len(self.input_sequences))
            if final_index>len(self.input_sequences):
                permuted_indices = permuted_indices[initial_index:]
            else:
                permuted_indices = permuted_indices[initial_index:final_index]
            return self.input_sequences[permuted_indices], self.output_targets[permuted_indices]


    def __len__(self):
        if self.is_train:
            return self.epoch_per_size
        else:
            # 确保至少有一个批次
            return max(1, self.num_per_epoch)