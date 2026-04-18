from typing import Optional, Union
import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from .components.ftd_dataset import FtdDataset


class FtdDataModule(LightningDataModule):
    """A class representing the FTD dataset for LSTM models.

    Attributes
    ----------
    train_size : float
        The proportion of the dataset to use for training.
    self.hparams.val_size : float
        The proportion of the dataset to use for validation.
    self.hparams.test_size : float
        The proportion of the dataset to use for testing.
    device : torch.device
        The device to use for computation (CPU or GPU).
    seq_length : int
        The length of input sequences for LSTM.

    Methods
    -------
    create_sequences(data, seq_length)
        Creates input sequences and corresponding targets for time series data.
    load_data(path, norm_method)
        Loads the dataset from the given path and normalizes it.
    split()
        Splits the dataset into training, validation, and testing sets.
    train_dataset()
        Returns the training dataset.
    val_dataset()
        Returns the validation dataset.
    test_dataset()
        Returns the testing dataset.
    """

    def __init__(
            self,
            data_dir: str = "data/",
            val_size: float = 0.1,
            test_size: float = 0.2,
            epoch_per_size: int = 1,
            batch_size: int = 1024,
            test_batch_size: int = 1024,
            num_workers: int = 8,
            pin_memory: bool = True,
            action: str = "",
            altitude: str = "",
            train_number: int = 100000,
            scaler=Union[StandardScaler, MinMaxScaler, None],
            seq_length: int = 10  # 新增序列长度参数
    ) -> None:
        """Initialize a new instance of the FtdDataset class for LSTM.

        Parameters
        ----------
        seq_length : int
            Length of input sequences for LSTM.
        """

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.scaler = scaler
        self.seq_length = seq_length  # 存储序列长度

    def create_sequences(self, data, targets):
        """Create input sequences and corresponding targets for time series data.

        :param data: Input data of shape (num_samples, num_features)
        :param targets: Target data of shape (num_samples, output_size)
        :return: Tuple of (sequences, targets)
                 sequences shape: (num_sequences, seq_length, num_features)
                 targets shape: (num_sequences, output_size)
        """
        sequences = []
        seq_targets = []
        for i in range(len(data) - self.seq_length):
            sequences.append(data[i:i + self.seq_length])
            seq_targets.append(targets[i:i + self.seq_length])
        return np.array(sequences), np.array(seq_targets)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train or self.data_val or self.data_test:
            return

        # 加载原始数据
        if not self.hparams.action and not self.hparams.altitude:
            (input_temp, _, input_test), (output_temp, _, output_test) = self.load_data()
        else:
            (input_temp, _, input_test), (output_temp, _, output_test) = self.load_data_lacking()

        # 划分训练和验证集
        input_train, input_val, output_train, output_val = train_test_split(
            input_temp, output_temp, test_size=self.hparams.val_size, random_state=42
        )

        # 归一化处理
        input_scaler = self.scaler
        input_train_norm = input_scaler.fit_transform(input_train)
        input_val_norm = input_scaler.transform(input_val)
        input_test_norm = input_scaler.transform(input_test)
        print("MinMaxScaler data range:")
        print("Min:", input_scaler.data_min_)
        print("Max:", input_scaler.data_max_)
        # 保存归一化参数用于后续反归一化
        np.save('input_scaler_min.npy', input_scaler.data_min_)
        np.save('input_scaler_max.npy', input_scaler.data_max_)

        output_scaler = self.scaler
        output_train_norm = output_scaler.fit_transform(output_train)
        output_val_norm = output_scaler.transform(output_val)
        output_test_norm = output_scaler.transform(output_test)
        print("MinMaxScaler data range:")
        print("Min:", output_scaler.data_min_)
        print("Max:", output_scaler.data_max_)


        np.save('output_scaler_min.npy', output_scaler.data_min_)
        np.save('output_scaler_max.npy', output_scaler.data_max_)

        # 创建序列数据
        X_train, y_train = self.create_sequences(input_train_norm, output_train_norm)
        X_val, y_val = self.create_sequences(input_val_norm, output_val_norm)
        X_test, y_test = self.create_sequences(input_test_norm, output_test_norm)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        print(f"Training sequences shape: {X_train.shape}, targets shape: {y_train.shape}")
        print(f"Validation sequences shape: {X_val.shape}, targets shape: {y_val.shape}")
        print(f"Testing sequences shape: {X_test.shape}, targets shape: {y_test.shape}")

        # 转换为Tensor
        self.data_train = FtdDataset(
            # torch.tensor(X_train, dtype=torch.float32),
            # torch.tensor(y_train, dtype=torch.float32),
            X_train, y_train,
            True,
            self.hparams.epoch_per_size,
            self.hparams.train_number
        )
        self.data_val = FtdDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            False,
            self.hparams.epoch_per_size,
            self.hparams.train_number
        )
        self.data_test = FtdDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
            False,
            self.hparams.epoch_per_size,
            self.hparams.train_number
        )

    # def load_data_lacking(
    #         self,
    # ) -> tuple[Dataset, Dataset, Dataset]:
    #     train_x = []
    #     train_y = []
    #     test_x = []
    #     test_y = []
    #     with h5py.File(self.hparams.data_dir, "r") as hdf5_file:
    #         for altitude_folder in hdf5_file:
    #             altitude_group = hdf5_file[altitude_folder]
    #             for action_folder in altitude_group:
    #                 action_group = altitude_group[action_folder]
    #                 if self.hparams.altitude:
    #                     if self.hparams.altitude == altitude_folder and (
    #                             self.hparams.action in (action_folder, "")
    #                     ):
    #                         test_x.append(action_group["x"][:])
    #                         test_y.append(action_group["y"][:])
    #                     else:
    #                         train_x.append(action_group["x"][:])
    #                         train_y.append(action_group["y"][:])
    #
    #                 elif self.hparams.action == action_folder:
    #                     test_x.append(action_group["x"][:])
    #                     test_y.append(action_group["y"][:])
    #                 else:
    #                     train_x.append(action_group["x"][:])
    #                     train_y.append(action_group["y"][:])
    #     hdf5_file.close()
    #     input_temp = np.vstack(train_x)
    #     output_temp = np.vstack(train_y)
    #     input_test = np.vstack(test_x)
    #     output_test = np.vstack(test_y)
    #     return (input_temp, None, input_test), (output_temp, None, output_test)

    def load_data_lacking(
            self,
    ) -> tuple[Dataset, Dataset, Dataset]:
        train_x = []
        train_y = []
        test_x = []
        test_y = []

        # --- 新增逻辑：将参数统一处理为列表 ---
        # 处理 altitude 参数：如果是字符串放入列表，如果是列表则保持，如果为空则为空列表
        target_altitudes = self.hparams.altitude
        if isinstance(target_altitudes, str) and target_altitudes:
            target_altitudes = [target_altitudes]
        elif not target_altitudes:
            target_altitudes = []
        # 确保如果是 tuple 等其他序列也转为 list
        elif isinstance(target_altitudes, (list, tuple)):
            target_altitudes = list(target_altitudes)

        # 处理 action 参数：同上
        target_actions = self.hparams.action
        if isinstance(target_actions, str) and target_actions:
            target_actions = [target_actions]
        elif not target_actions:
            target_actions = []
        elif isinstance(target_actions, (list, tuple)):
            target_actions = list(target_actions)
        # ---------------------------------------

        with h5py.File(self.hparams.data_dir, "r") as hdf5_file:
            for altitude_folder in hdf5_file:
                altitude_group = hdf5_file[altitude_folder]
                for action_folder in altitude_group:
                    action_group = altitude_group[action_folder]

                    is_test_data = False

                    # --- 修改后的判断逻辑 ---
                    if target_altitudes:
                        # 情况1：指定了高度列表
                        # 只有当前高度在目标列表中时，才考虑作为测试集
                        if altitude_folder in target_altitudes:
                            if target_actions:
                                # 如果也指定了动作，必须动作也匹配才作为测试集
                                if action_folder in target_actions:
                                    is_test_data = True
                            else:
                                # 如果只指定了高度没指定动作，该高度下所有数据都作为测试集
                                is_test_data = True
                    elif target_actions:
                        # 情况2：没指定高度，但指定了动作列表
                        # 所有高度下的该动作都作为测试集
                        if action_folder in target_actions:
                            is_test_data = True
                    # -----------------------

                    if is_test_data:
                        test_x.append(action_group["x"][:])
                        test_y.append(action_group["y"][:])
                    else:
                        train_x.append(action_group["x"][:])
                        train_y.append(action_group["y"][:])

        hdf5_file.close()
        input_temp = np.vstack(train_x)
        output_temp = np.vstack(train_y)
        input_test = np.vstack(test_x)
        output_test = np.vstack(test_y)
        input_train, input_val, output_train, output_val = train_test_split(
            input_temp,
            output_temp,
            test_size=self.hparams.val_size,
        )
        return (input_train, input_val, input_test), (
            output_train,
            output_val,
            output_test,
        )



    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        xs = []
        ys = []
        with h5py.File(self.hparams.data_dir, "r") as hdf5_file:
            for altitude_folder in hdf5_file:
                altitude_group = hdf5_file[altitude_folder]
                for action_folder in altitude_group:
                    action_group = altitude_group[action_folder]
                    xs.append(action_group["x"][:])
                    ys.append(action_group["y"][:])
        hdf5_file.close()
        x = np.vstack(xs)
        y = np.vstack(ys)
        input_temp, input_test, output_temp, output_test = train_test_split(
            x, y, test_size=self.hparams.test_size, random_state=42
        )
        return (input_temp, None, input_test), (output_temp, None, output_test)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader for sequences.

        Returns
        -------
        DataLoader
            Dataloader for training sequences and targets.
        """
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,  # 对于序列数据，建议shuffle
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader for sequences.

        Returns
        -------
        DataLoader
            Dataloader for validation sequences and targets.
        """
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the testing dataloader for sequences.

        Returns
        -------
        DataLoader
            Dataloader for testing sequences and targets.
        """
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


if __name__ == "__main__":
    path = "/harddisk/Dataset/ftd_dataset/basedata/&H10000F0V211X21/BTB"
    datamodule = FtdDataModule(
        data_dir=path,
        scaler=MinMaxScaler(),
        seq_length=10  # 测试序列长度
    )
    datamodule.setup()

    # 测试数据加载
    train_loader = datamodule.train_dataloader()
    for batch_idx, (x, y) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Input shape: {x.shape}, Target shape: {y.shape}")
        if batch_idx > 2:  # 只查看前几个批次
            break