from typing import Any, Dict, Tuple, Optional, List, Iterator
from itertools import chain
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric
from src.models.components.mlp import MLP
from src.models.components.weight_methods import WeightMethods


class MultiTaskLSTMWithEncoderDecoder(nn.Module):
    """结合编码器-解码器和多任务LSTM的模型"""

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            task_sizes: list,  # 每个任务的输出大小
            task_names: list = None,
            # 编码器-解码器参数
            encoder_dims: list = None,  # 编码器维度 [input_size, hidden1, hidden2, ...]
            decoder_dims: list = None,  # 解码器基础维度，每个任务会单独创建解码器
            # LSTM参数
            dropout: float = 0.0,
            shared_layers: int = 2,  # 共享LSTM层数
            task_specific_layers: int = 1,  # 任务特定层数
            # 编码器参数
            encoder_batch_norm: bool = False,
            # 解码器参数
            decoder_batch_norm: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task_sizes = task_sizes
        self.num_tasks = len(task_sizes)
        self.task_names = task_names or [f"task_{i}" for i in range(self.num_tasks)]
        self.shared_layers = shared_layers
        self.task_specific_layers = task_specific_layers

        # 设置编码器维度
        if encoder_dims is None:
            encoder_dims = [input_size, 512, 512, 64]
        self.encoder_dims = encoder_dims

        # 设置解码器基础维度 - 确保第一层与LSTM hidden_size匹配
        if decoder_dims is None:
            decoder_dims = [hidden_size, 512, 512]  # 第一层改为hidden_size
        else:
            # 确保解码器第一层维度与LSTM输出匹配
            decoder_dims = [hidden_size] + decoder_dims[1:]

        # 编码器：将输入映射到潜在空间
        self.encoder = MLP(
            channel_list=encoder_dims,
            batch_norm=encoder_batch_norm,
            dropout=dropout
        )

        # 共享的LSTM层 - 处理编码后的序列
        lstm_input_size = encoder_dims[-1]  # 编码器输出维度作为LSTM输入
        self.shared_lstm = nn.LSTM(
            lstm_input_size,
            hidden_size,
            shared_layers,
            batch_first=True,
            dropout=dropout if shared_layers > 1 else 0.0
        )

        # 任务特定的LSTM层
        self.task_specific_lstms = nn.ModuleDict()
        for task_name in self.task_names:
            task_lstm = nn.LSTM(
                hidden_size,
                hidden_size,
                task_specific_layers,
                batch_first=True,
                dropout=dropout if task_specific_layers > 1 else 0.0
            )
            self.task_specific_lstms[task_name] = task_lstm

        # 任务特定的解码器
        self.task_decoders = nn.ModuleDict()
        for i, (task_name, task_size) in enumerate(zip(self.task_names, self.task_sizes)):
            # 每个任务有自己的解码器
            decoder_dims_with_output = decoder_dims + [task_size]
            task_decoder = MLP(
                channel_list=decoder_dims_with_output,
                batch_norm=decoder_batch_norm,
                dropout=dropout
            )
            self.task_decoders[task_name] = task_decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入序列 [batch, seq_len, input_size] 或 [1, batch, seq_len, input_size]

        Returns:
            所有任务的输出拼接 [batch, seq_len, sum(task_sizes)]
        """
        # 处理4D输入
        if x.dim() == 4:
            if x.size(0) == 1:
                x = x.squeeze(0)
            else:
                raise ValueError(f"不支持的输入形状: {x.shape}")

        batch_size, seq_len, input_dim = x.shape

        # 编码器处理每个时间步
        encoded_sequence = []
        for t in range(seq_len):
            time_step = x[:, t, :]  # [batch, input_size]
            encoded_time_step = self.encoder(time_step)  # [batch, encoder_output_size]
            encoded_sequence.append(encoded_time_step.unsqueeze(1))

        encoded_sequence = torch.cat(encoded_sequence, dim=1)  # [batch, seq_len, encoder_output_size]

        # 共享LSTM处理编码后的序列
        lstm_output, (hidden, cell) = self.shared_lstm(encoded_sequence)

        # 各任务解码器处理
        task_outputs = []
        for task_name in self.task_names:
            task_decoder = self.task_decoders[task_name]

            # 对每个时间步应用任务解码器
            task_time_outputs = []
            for t in range(seq_len):
                time_step_lstm_output = lstm_output[:, t, :]  # [batch, hidden_size]
                task_time_output = task_decoder(time_step_lstm_output)  # [batch, task_size]
                task_time_outputs.append(task_time_output.unsqueeze(1))

            task_output = torch.cat(task_time_outputs, dim=1)  # [batch, seq_len, task_size]
            task_outputs.append(task_output)

        # 拼接所有任务的输出
        return torch.cat(task_outputs, dim=-1)

    # ... 其他方法保持不变 ...

    def get_task_output(self, x: torch.Tensor, task_index: int) -> torch.Tensor:
        """获取特定任务的输出"""
        if task_index >= self.num_tasks:
            raise ValueError(f"任务索引 {task_index} 超出范围，共有 {self.num_tasks} 个任务")

        # 处理输入
        if x.dim() == 4:
            x = x.squeeze(0)

        batch_size, seq_len, input_dim = x.shape
        task_name = self.task_names[task_index]

        # 编码器处理
        encoded_sequence = []
        for t in range(seq_len):
            time_step = x[:, t, :]
            encoded_time_step = self.encoder(time_step)
            encoded_sequence.append(encoded_time_step.unsqueeze(1))
        encoded_sequence = torch.cat(encoded_sequence, dim=1)

        # LSTM处理
        lstm_output, (hidden, cell) = self.shared_lstm(encoded_sequence)

        # 特定任务解码
        task_decoder = self.task_decoders[task_name]
        task_time_outputs = []
        for t in range(seq_len):
            time_step_lstm_output = lstm_output[:, t, :]
            task_time_output = task_decoder(time_step_lstm_output)
            task_time_outputs.append(task_time_output.unsqueeze(1))

        return torch.cat(task_time_outputs, dim=1)

    def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        """共享参数：编码器和共享LSTM"""
        return chain(
            self.encoder.parameters(),
            self.shared_lstm.parameters()
        )

    def task_specific_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        """任务特定参数：各任务的解码器"""
        return chain(*[decoder.parameters() for decoder in self.task_decoders.values()])

    def last_shared_parameters(self):
        """最后共享层参数：LSTM最后一层"""
        return list(self.shared_lstm.parameters())[-2:]  # 最后一层的权重和偏置


class FtdModule(LightningModule):
    def __init__(
            self,
            # LSTM 参数
            input_size: int,
            hidden_size: int,
            num_layers: int,
            # 多任务参数
            task_sizes: List[int],  # 每个任务的输出维度
            task_names: List[str] = None,  # 任务名称
            # 编码器-解码器参数
            encoder_dims: List[int] = None,  # 编码器维度
            decoder_dims: List[int] = None,  # 解码器基础维度
            # 模型结构参数
            shared_layers: int = 2,
            task_specific_layers: int = 1,
            # 训练参数
            dropout: float = 0.0,
            learning_rate: float = 0.001,
            loss_term: str = "",  # 多任务学习方法
            encoder_batch_norm: bool = False,
            decoder_batch_norm: bool = False,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # 多任务LSTM模型（带编码器-解码器）
        self.net = MultiTaskLSTMWithEncoderDecoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            task_sizes=task_sizes,
            task_names=task_names,
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims,
            dropout=dropout,
            shared_layers=shared_layers,
            task_specific_layers=task_specific_layers,
            encoder_batch_norm=encoder_batch_norm,
            decoder_batch_norm=decoder_batch_norm,
        )

        # 损失函数和指标
        self.criterion = torch.nn.MSELoss(reduction="none")

        # 训练指标 - 使用 nn.ModuleList 确保设备同步
        self.train_loss = MeanMetric()
        self.train_task_losses = nn.ModuleList([MeanMetric() for _ in range(len(task_sizes))])

        # 验证指标
        self.val_loss = MeanMetric()
        self.val_task_losses = nn.ModuleList([MeanMetric() for _ in range(len(task_sizes))])

        # 测试指标
        self.test_loss = MeanMetric()
        self.test_task_losses = nn.ModuleList([MeanMetric() for _ in range(len(task_sizes))])

        # 存储预测结果
        self.saved_x = None
        self.saved_y = None
        self.saved_preds = None

        # 多任务学习相关
        self.loss_term = loss_term
        self.task_num = len(task_sizes)
        self.automatic_optimization = False

    def on_fit_start(self):
        """确保所有指标在训练开始时都在正确的设备上"""
        # 手动将指标移动到当前设备
        device = self.device
        self.train_loss = self.train_loss.to(device)
        self.val_loss = self.val_loss.to(device)

        for i in range(len(self.train_task_losses)):
            self.train_task_losses[i] = self.train_task_losses[i].to(device)
            self.val_task_losses[i] = self.val_task_losses[i].to(device)
            self.test_task_losses[i] = self.test_task_losses[i].to(device)

    def on_test_start(self):
        """确保所有指标在测试开始时都在正确的设备上"""
        device = self.device
        self.test_loss = self.test_loss.to(device)
        for i in range(len(self.test_task_losses)):
            self.test_task_losses[i] = self.test_task_losses[i].to(device)
    def setup(self, stage: str) -> None:
        """设置多任务学习方法"""
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

        if stage == "fit" and self.loss_term in ["stch", "fairgrad", "famo"]:
            self.weight_method = WeightMethods(
                method=self.loss_term,
                n_tasks=self.task_num,
                device=self.device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 处理4D输入 [1, batch, seq, features]
        if x.dim() == 4:
            if x.size(0) == 1:
                x = x.squeeze(0)
            else:
                raise ValueError(f"不支持的输入形状: {x.shape}")
        return self.net(x)

    def get_task_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """获取每个任务的独立预测"""
        task_predictions = []
        for i in range(self.task_num):
            task_pred = self.net.get_task_output(x, i)
            task_predictions.append(task_pred)
        return task_predictions

    def compute_task_losses(self, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算每个任务的损失"""
        batch_size, seq_len, total_output_size = preds.shape

        # 计算每个任务的损失
        task_losses = []
        start_idx = 0

        for i, task_size in enumerate(self.hparams.task_sizes):
            end_idx = start_idx + task_size
            task_pred = preds[:, :, start_idx:end_idx]
            task_target = y[:, :, start_idx:end_idx]

            # 计算该任务的MSE损失并平均
            task_loss = self.criterion(task_pred, task_target).mean()
            task_losses.append(task_loss)

            start_idx = end_idx

        return torch.stack(task_losses)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        for metric in self.val_task_losses:
            metric.reset()

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        # 处理输入维度
        x = x.squeeze(0) if x.dim() == 4 else x
        y = y.squeeze(0) if y.dim() == 4 else y

        # 前向传播
        preds = self.forward(x)

        # 优化器
        opt = self.optimizers()
        opt.zero_grad()

        # 计算每个任务的损失
        task_losses = self.compute_task_losses(preds, y)

        if self.loss_term in ["stch", "fairgrad", "famo"]:
            # 使用多任务学习方法
            _, extra_outputs = self.weight_method.backward(
                losses=task_losses,
                shared_parameters=list(self.net.shared_parameters()),
                task_specific_parameters=list(self.net.task_specific_parameters()),
                last_shared_parameters=list(self.net.last_shared_parameters()),
                epoch=self.current_epoch,
            )
            total_loss = task_losses.mean()
        else:
            # 普通训练：简单求和
            total_loss = task_losses.sum()
            self.manual_backward(total_loss)

        opt.step()

        # FAMO 更新
        if "famo" in self.loss_term:
            with torch.no_grad():
                preds = self.forward(x)
                task_losses_new = self.compute_task_losses(preds, y)
                self.weight_method.method.update(task_losses_new.detach())

        # 记录指标
        self.train_loss(total_loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        # 记录每个任务的损失
        for i, task_loss in enumerate(task_losses):
            task_name = self.net.task_names[i] if i < len(self.net.task_names) else f"task_{i}"
            self.log(f"train/{task_name}_loss", task_loss, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    def validation_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch

        # 处理输入维度
        x = x.squeeze(0) if x.dim() == 4 else x
        y = y.squeeze(0) if y.dim() == 4 else y

        # 前向传播
        preds = self.forward(x)

        # 计算损失
        task_losses = self.compute_task_losses(preds, y)
        total_loss = task_losses.mean()

        # 记录指标
        self.val_loss(total_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 记录每个任务的验证损失
        for i, task_loss in enumerate(task_losses):
            task_name = self.net.task_names[i] if i < len(self.net.task_names) else f"task_{i}"
            self.val_task_losses[i](task_loss)
            self.log(f"val/{task_name}_loss", self.val_task_losses[i], on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch

        # 处理输入维度
        x = x.squeeze(0) if x.dim() == 4 else x
        y = y.squeeze(0) if y.dim() == 4 else y

        # 前向传播
        preds = self.forward(x)

        # 计算损失
        task_losses = self.compute_task_losses(preds, y)
        total_loss = task_losses.mean()

        # 记录指标
        self.test_loss(total_loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 记录每个任务的测试损失
        for i, task_loss in enumerate(task_losses):
            task_name = self.net.task_names[i] if i < len(self.net.task_names) else f"task_{i}"
            self.test_task_losses[i](task_loss)
            self.log(f"test/{task_name}_loss", self.test_task_losses[i], on_step=False, on_epoch=True, prog_bar=True)

        # 存储预测结果
        self.saved_x = x.detach()
        self.saved_y = y.detach()
        self.saved_preds = preds.detach()

    def on_test_epoch_end(self) -> None:
        # 保存数据
        if self.saved_x is not None:
            torch.save(self.saved_x, "saved_x.pth")
            torch.save(self.saved_y, "saved_y.pth")
            torch.save(self.saved_preds, "saved_preds.pth")

    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和调度器"""
        # 优化器配置
        if hasattr(self, 'hparams') and hasattr(self.hparams, 'optimizer') and self.hparams.optimizer is not None:
            if isinstance(self.hparams.optimizer, dict):
                optimizer_class = self.hparams.optimizer.get('_target_', torch.optim.Adam)
                optimizer_params = {k: v for k, v in self.hparams.optimizer.items()
                                    if k != '_target_' and k != 'norm_layer'}
                optimizer_params['params'] = self.parameters()
                optimizer = optimizer_class(**optimizer_params)
            elif callable(self.hparams.optimizer):
                optimizer = self.hparams.optimizer(params=self.parameters())
            else:
                lr = getattr(self, 'learning_rate', getattr(self.hparams, 'learning_rate', 0.001))
                optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            lr = getattr(self, 'learning_rate', getattr(self.hparams, 'learning_rate', 0.001))
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # 调度器配置
        if hasattr(self, 'hparams') and hasattr(self.hparams, 'scheduler') and self.hparams.scheduler is not None:
            scheduler = None
            if isinstance(self.hparams.scheduler, dict):
                scheduler_class = self.hparams.scheduler.get('_target_', None)
                if scheduler_class is not None:
                    valid_params = ['optimizer', 'mode', 'factor', 'patience', 'verbose',
                                    'threshold', 'threshold_mode', 'cooldown', 'min_lr', 'eps']
                    scheduler_params = {k: v for k, v in self.hparams.scheduler.items()
                                        if k != '_target_' and k in valid_params}
                    scheduler_params['optimizer'] = optimizer
                    try:
                        scheduler = scheduler_class(**scheduler_params)
                    except TypeError as e:
                        print(f"Error creating scheduler: {e}")
                        scheduler = None

            if scheduler is not None:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }

        return {"optimizer": optimizer}


if __name__ == "__main__":
    # 测试代码
    model = FtdModule(
        input_size=31,
        hidden_size=128,
        num_layers=4,
        task_sizes=[1, 1, 1, 1, 1, 1],
        task_names=["Cy", "Cl", "Cm", "Cn", "CD", "CL"],
        encoder_dims=[31, 512, 512, 64],
        decoder_dims=[64, 512, 512],
        loss_term="stch"
    )

    # 测试前向传播
    test_input = torch.randn(1, 32, 10, 31)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

# # from typing import Any, Dict, Tuple, Optional, List, Iterator
# # from itertools import chain
# # import torch
# # import torch.nn as nn
# # from lightning import LightningModule
# # from torchmetrics import MeanMetric
# # from src.models.components.mlp import MLP
# # from src.models.components.weight_methods import WeightMethods
# #
# #
# # class MultiTaskLSTMWithEncoderDecoder(nn.Module):
# #     """结合编码器-解码器和多任务LSTM的模型"""
# #
# #     def __init__(
# #             self,
# #             input_size: int,
# #             hidden_size: int,
# #             task_sizes: list,  # 每个任务(分组)的输出大小
# #             task_names: list = None,
# #             # 编码器-解码器参数
# #             encoder_dims: list = None,
# #             decoder_dims: list = None,
# #             # LSTM参数
# #             dropout: float = 0.0,
# #             shared_layers: int = 2,
# #             task_specific_layers: int = 1,
# #             # 编码器参数
# #             encoder_batch_norm: bool = False,
# #             # 解码器参数
# #             decoder_batch_norm: bool = False,
# #     ):
# #         super().__init__()
# #
# #         self.input_size = input_size
# #         self.hidden_size = hidden_size
# #         self.task_sizes = task_sizes
# #         self.num_tasks = len(task_sizes)
# #         self.task_names = task_names or [f"task_{i}" for i in range(self.num_tasks)]
# #         self.shared_layers = shared_layers
# #         self.task_specific_layers = task_specific_layers
# #
# #         # 设置编码器维度
# #         if encoder_dims is None:
# #             encoder_dims = [input_size, 512, 512, 64]
# #         self.encoder_dims = encoder_dims
# #
# #         # 设置解码器基础维度
# #         if decoder_dims is None:
# #             decoder_dims = [hidden_size, 512, 512]
# #         else:
# #             decoder_dims = [hidden_size] + decoder_dims[1:]
# #
# #         # 编码器
# #         self.encoder = MLP(
# #             channel_list=encoder_dims,
# #             batch_norm=encoder_batch_norm,
# #             dropout=dropout
# #         )
# #
# #         # 共享LSTM
# #         lstm_input_size = encoder_dims[-1]
# #         self.shared_lstm = nn.LSTM(
# #             lstm_input_size,
# #             hidden_size,
# #             shared_layers,
# #             batch_first=True,
# #             dropout=dropout if shared_layers > 1 else 0.0
# #         )
# #
# #         # 任务特定LSTM
# #         self.task_specific_lstms = nn.ModuleDict()
# #         for task_name in self.task_names:
# #             task_lstm = nn.LSTM(
# #                 hidden_size,
# #                 hidden_size,
# #                 task_specific_layers,
# #                 batch_first=True,
# #                 dropout=dropout if task_specific_layers > 1 else 0.0
# #             )
# #             self.task_specific_lstms[task_name] = task_lstm
# #
# #         # 任务特定解码器
# #         self.task_decoders = nn.ModuleDict()
# #         for i, (task_name, task_size) in enumerate(zip(self.task_names, self.task_sizes)):
# #             decoder_dims_with_output = decoder_dims + [task_size]
# #             task_decoder = MLP(
# #                 channel_list=decoder_dims_with_output,
# #                 batch_norm=decoder_batch_norm,
# #                 dropout=dropout
# #             )
# #             self.task_decoders[task_name] = task_decoder
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         """前向传播: 返回拼接后的所有任务输出"""
# #         if x.dim() == 4:
# #             x = x.squeeze(0) if x.size(0) == 1 else x
# #
# #         batch_size, seq_len, _ = x.shape
# #
# #         # Encoder
# #         encoded_sequence = []
# #         for t in range(seq_len):
# #             encoded_sequence.append(self.encoder(x[:, t, :]).unsqueeze(1))
# #         encoded_sequence = torch.cat(encoded_sequence, dim=1)
# #
# #         # Shared LSTM
# #         lstm_output, _ = self.shared_lstm(encoded_sequence)
# #
# #         # Task Specific Decoders
# #         task_outputs = []
# #         for task_name in self.task_names:
# #             task_decoder = self.task_decoders[task_name]
# #             task_time_outputs = []
# #             for t in range(seq_len):
# #                 task_time_outputs.append(task_decoder(lstm_output[:, t, :]).unsqueeze(1))
# #             task_outputs.append(torch.cat(task_time_outputs, dim=1))
# #
# #         return torch.cat(task_outputs, dim=-1)
# #
# #     def get_task_output(self, x: torch.Tensor, task_index: int) -> torch.Tensor:
# #         if task_index >= self.num_tasks:
# #             raise ValueError(f"Task index {task_index} out of range")
# #
# #         if x.dim() == 4: x = x.squeeze(0)
# #         batch_size, seq_len, _ = x.shape
# #         task_name = self.task_names[task_index]
# #
# #         encoded_sequence = []
# #         for t in range(seq_len):
# #             encoded_sequence.append(self.encoder(x[:, t, :]).unsqueeze(1))
# #         encoded_sequence = torch.cat(encoded_sequence, dim=1)
# #
# #         lstm_output, _ = self.shared_lstm(encoded_sequence)
# #
# #         task_decoder = self.task_decoders[task_name]
# #         task_time_outputs = []
# #         for t in range(seq_len):
# #             task_time_outputs.append(task_decoder(lstm_output[:, t, :]).unsqueeze(1))
# #
# #         return torch.cat(task_time_outputs, dim=1)
# #
# #     def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
# #         return chain(self.encoder.parameters(), self.shared_lstm.parameters())
# #
# #     def task_specific_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
# #         return chain(*[decoder.parameters() for decoder in self.task_decoders.values()])
# #
# #     def last_shared_parameters(self):
# #         return list(self.shared_lstm.parameters())[-2:]
# #
# #
# # class FtdModule(LightningModule):
# #     def __init__(
# #             self,
# #             input_size: int,
# #             hidden_size: int,
# #             task_sizes: List[int],  # 分组后的输出维度，如 [3, 3]
# #             task_names: List[str] = None,  # 分组名称，如 ["Lateral", "Longitudinal"]
# #             task_indices: List[List[int]] = None,  # 索引映射，如 [[0,1,3], [2,4,5]]
# #
# #             # 【新增】原始任务名称，用于显示详细损失
# #             original_task_names: List[str] = None,  # 如 ["Cy", "Cl", "Cm", "Cn", "CD", "CL"]
# #
# #             encoder_dims: List[int] = None,
# #             decoder_dims: List[int] = None,
# #             shared_layers: int = 2,
# #             task_specific_layers: int = 1,
# #             dropout: float = 0.0,
# #             learning_rate: float = 0.001,
# #             loss_term: str = "",
# #             encoder_batch_norm: bool = False,
# #             decoder_batch_norm: bool = False,
# #             optimizer: Optional[torch.optim.Optimizer] = None,
# #             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
# #             compile: bool = False,
# #     ) -> None:
# #         super().__init__()
# #         self.save_hyperparameters()
# #
# #         # 处理 task_indices
# #         if task_indices is None:
# #             current_idx = 0
# #             self.task_indices = []
# #             for size in task_sizes:
# #                 self.task_indices.append(list(range(current_idx, current_idx + size)))
# #                 current_idx += size
# #         else:
# #             self.task_indices = task_indices
# #
# #         # 保存原始任务名，如果没有提供则默认为 indices 的字符串
# #         self.original_task_names = original_task_names
# #
# #         # 模型
# #         self.net = MultiTaskLSTMWithEncoderDecoder(
# #             input_size=input_size,
# #             hidden_size=hidden_size,
# #             task_sizes=task_sizes,
# #             task_names=task_names,
# #             encoder_dims=encoder_dims,
# #             decoder_dims=decoder_dims,
# #             dropout=dropout,
# #             shared_layers=shared_layers,
# #             task_specific_layers=task_specific_layers,
# #             encoder_batch_norm=encoder_batch_norm,
# #             decoder_batch_norm=decoder_batch_norm,
# #         )
# #
# #         self.criterion = torch.nn.MSELoss(reduction="none")
# #
# #         # --- 指标定义 ---
# #         # 1. 总损失
# #         self.train_loss = MeanMetric()
# #         self.val_loss = MeanMetric()
# #         self.test_loss = MeanMetric()
# #
# #         # 2. 分组损失 (用于优化器和粗粒度监控)
# #         self.train_group_losses = nn.ModuleList([MeanMetric() for _ in range(len(task_sizes))])
# #         self.val_group_losses = nn.ModuleList([MeanMetric() for _ in range(len(task_sizes))])
# #         self.test_group_losses = nn.ModuleList([MeanMetric() for _ in range(len(task_sizes))])
# #
# #         # 3. 【新增】详细分量损失 (用于细粒度监控)
# #         # 使用 ModuleDict 方便按名字管理，并确保自动移动到 GPU
# #         if self.original_task_names:
# #             self.train_detail_losses = nn.ModuleDict({name: MeanMetric() for name in self.original_task_names})
# #             self.val_detail_losses = nn.ModuleDict({name: MeanMetric() for name in self.original_task_names})
# #             self.test_detail_losses = nn.ModuleDict({name: MeanMetric() for name in self.original_task_names})
# #         else:
# #             # 如果没有提供名字，暂不创建或使用默认名
# #             self.train_detail_losses = nn.ModuleDict()
# #             self.val_detail_losses = nn.ModuleDict()
# #             self.test_detail_losses = nn.ModuleDict()
# #
# #         # 其他
# #         self.saved_x = None;
# #         self.saved_y = None;
# #         self.saved_preds = None
# #         self.loss_term = loss_term
# #         self.task_num = len(task_sizes)
# #         self.automatic_optimization = False
# #
# #     def setup(self, stage: str) -> None:
# #         if self.hparams.compile and stage == "fit":
# #             self.net = torch.compile(self.net)
# #         if stage == "fit" and self.loss_term in ["stch", "fairgrad", "famo"]:
# #             self.weight_method = WeightMethods(self.loss_term, self.task_num, self.device)
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         return self.net(x)
# #
# #     def compute_task_losses(self, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
# #         """计算分组损失 (用于优化)"""
# #         task_losses = []
# #         pred_start_idx = 0
# #         for i, task_size in enumerate(self.hparams.task_sizes):
# #             pred_end_idx = pred_start_idx + task_size
# #             task_pred = preds[:, :, pred_start_idx:pred_end_idx]
# #             task_target = y[:, :, self.task_indices[i]]  # 使用索引映射
# #             task_losses.append(self.criterion(task_pred, task_target).mean())
# #             pred_start_idx = pred_end_idx
# #         return torch.stack(task_losses)
# #
# #     def compute_and_log_detailed_losses(self, preds: torch.Tensor, y: torch.Tensor, stage: str):
# #         """【新增】计算并记录详细的单分量损失"""
# #         if not self.original_task_names:
# #             return
# #
# #         metrics_dict = getattr(self, f"{stage}_detail_losses")
# #         pred_ptr = 0
# #
# #         # 遍历每个分组
# #         for group_idx, indices_in_group in enumerate(self.task_indices):
# #             # 遍历分组内的每个分量
# #             for i, original_idx in enumerate(indices_in_group):
# #                 # 提取单个分量的预测值 (preds 是按分组顺序拼接的)
# #                 # 分组起始位置 pred_ptr + 当前分量在组内的偏移 i
# #                 pred_slice = preds[:, :, pred_ptr + i: pred_ptr + i + 1]
# #
# #                 # 提取单个分量的真实值 (y 是原始顺序)
# #                 target_slice = y[:, :, original_idx: original_idx + 1]
# #
# #                 # 计算损失
# #                 loss = self.criterion(pred_slice, target_slice).mean()
# #
# #                 # 记录日志
# #                 if original_idx < len(self.original_task_names):
# #                     name = self.original_task_names[original_idx]
# #                     if name in metrics_dict:
# #                         metrics_dict[name](loss)
# #                         self.log(f"{stage}/detail_{name}_loss", metrics_dict[name], on_step=False, on_epoch=True,
# #                                  prog_bar=False)
# #
# #             # 更新分组指针
# #             pred_ptr += len(indices_in_group)
# #
# #     def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
# #         x, y = batch
# #         x = x.squeeze(0) if x.dim() == 4 else x
# #         y = y.squeeze(0) if y.dim() == 4 else y
# #
# #         preds = self.forward(x)
# #         opt = self.optimizers()
# #         opt.zero_grad()
# #
# #         # 1. 计算分组损失 (用于优化)
# #         group_losses = self.compute_task_losses(preds, y)
# #
# #         # 2. 多任务加权
# #         if self.loss_term in ["stch", "fairgrad", "famo"]:
# #             _, extra = self.weight_method.backward(
# #                 losses=group_losses,
# #                 shared_parameters=list(self.net.shared_parameters()),
# #                 task_specific_parameters=list(self.net.task_specific_parameters()),
# #                 last_shared_parameters=list(self.net.last_shared_parameters()),
# #                 epoch=self.current_epoch,
# #             )
# #             total_loss = group_losses.mean()
# #         else:
# #             total_loss = group_losses.sum()
# #             self.manual_backward(total_loss)
# #
# #         opt.step()
# #
# #         # FAMO update
# #         if "famo" in self.loss_term:
# #             with torch.no_grad():
# #                 preds_new = self.forward(x)
# #                 group_losses_new = self.compute_task_losses(preds_new, y)
# #                 self.weight_method.method.update(group_losses_new.detach())
# #
# #         # 3. 日志记录
# #         # 总损失
# #         self.train_loss(total_loss)
# #         self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
# #
# #         # 分组损失
# #         for i, loss in enumerate(group_losses):
# #             name = self.net.task_names[i]
# #             self.train_group_losses[i](loss)
# #             self.log(f"train/group_{name}_loss", self.train_group_losses[i], on_step=True, on_epoch=True,
# #                      prog_bar=False)
# #
# #         # 【新增】详细损失
# #         with torch.no_grad():
# #             self.compute_and_log_detailed_losses(preds, y, "train")
# #
# #         return total_loss
# #
# #     def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
# #         x, y = batch
# #         x = x.squeeze(0) if x.dim() == 4 else x
# #         y = y.squeeze(0) if y.dim() == 4 else y
# #
# #         preds = self.forward(x)
# #         group_losses = self.compute_task_losses(preds, y)
# #         total_loss = group_losses.mean()
# #
# #         self.val_loss(total_loss)
# #         self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
# #
# #         for i, loss in enumerate(group_losses):
# #             name = self.net.task_names[i]
# #             self.val_group_losses[i](loss)
# #             self.log(f"val/group_{name}_loss", self.val_group_losses[i], on_step=False, on_epoch=True, prog_bar=True)
# #
# #         # 【新增】详细损失
# #         self.compute_and_log_detailed_losses(preds, y, "val")
# #
# #     def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
# #         x, y = batch
# #         x = x.squeeze(0) if x.dim() == 4 else x
# #         y = y.squeeze(0) if y.dim() == 4 else y
# #
# #         preds = self.forward(x)
# #         group_losses = self.compute_task_losses(preds, y)
# #         total_loss = group_losses.mean()
# #
# #         self.test_loss(total_loss)
# #         self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
# #
# #         for i, loss in enumerate(group_losses):
# #             name = self.net.task_names[i]
# #             self.test_group_losses[i](loss)
# #             self.log(f"test/group_{name}_loss", self.test_group_losses[i], on_step=False, on_epoch=True, prog_bar=True)
# #
# #         self.compute_and_log_detailed_losses(preds, y, "test")
# #
# #         self.saved_x = x.detach();
# #         self.saved_y = y.detach();
# #         self.saved_preds = preds.detach()
# #
# #     def on_test_epoch_end(self) -> None:
# #         if self.saved_x is not None:
# #             torch.save(self.saved_x, "saved_x.pth")
# #             torch.save(self.saved_y, "saved_y.pth")
# #             torch.save(self.saved_preds, "saved_preds.pth")
# #
# #     # ... configure_optimizers 保持不变，省略以节省空间 ...
# #     def configure_optimizers(self) -> Dict[str, Any]:
# #         """配置优化器和调度器"""
# #         if hasattr(self, 'hparams') and hasattr(self.hparams, 'optimizer') and self.hparams.optimizer is not None:
# #             if isinstance(self.hparams.optimizer, dict):
# #                 optimizer_class = self.hparams.optimizer.get('_target_', torch.optim.Adam)
# #                 optimizer_params = {k: v for k, v in self.hparams.optimizer.items() if k != '_target_'}
# #                 optimizer_params['params'] = self.parameters()
# #                 optimizer = optimizer_class(**optimizer_params)
# #             elif callable(self.hparams.optimizer):
# #                 optimizer = self.hparams.optimizer(params=self.parameters())
# #             else:
# #                 lr = getattr(self, 'learning_rate', getattr(self.hparams, 'learning_rate', 0.001))
# #                 optimizer = torch.optim.Adam(self.parameters(), lr=lr)
# #         else:
# #             lr = getattr(self, 'learning_rate', getattr(self.hparams, 'learning_rate', 0.001))
# #             optimizer = torch.optim.Adam(self.parameters(), lr=lr)
# #
# #         if hasattr(self, 'hparams') and hasattr(self.hparams, 'scheduler') and self.hparams.scheduler is not None:
# #             scheduler = None
# #             if isinstance(self.hparams.scheduler, dict):
# #                 scheduler_class = self.hparams.scheduler.get('_target_', None)
# #                 if scheduler_class is not None:
# #                     valid_params = ['optimizer', 'mode', 'factor', 'patience', 'verbose', 'threshold', 'threshold_mode',
# #                                     'cooldown', 'min_lr', 'eps']
# #                     scheduler_params = {k: v for k, v in self.hparams.scheduler.items() if
# #                                         k != '_target_' and k in valid_params}
# #                     scheduler_params['optimizer'] = optimizer
# #                     try:
# #                         scheduler = scheduler_class(**scheduler_params)
# #                     except TypeError:
# #                         scheduler = None
# #             if scheduler is not None:
# #                 return {"optimizer": optimizer,
# #                         "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss", "interval": "epoch",
# #                                          "frequency": 1}}
# #         return {"optimizer": optimizer}
# #
# #
# # if __name__ == "__main__":
# #     # 测试代码更新
# #     model = FtdModule(
# #         input_size=31,
# #         hidden_size=128,
# #         task_sizes=[3, 3],
# #         task_names=["Lateral", "Longitudinal"],  # 分组名称
# #         task_indices=[[0, 1, 3], [2, 4, 5]],  # 映射索引
# #
# #         # 【新增】原始6个任务名称
# #         original_task_names=["Cy", "Cl", "Cm", "Cn", "CD", "CL"],
# #
# #         encoder_dims=[31, 512, 512, 64],
# #         decoder_dims=[64, 512, 512],
# #         loss_term="stch"
# #     )
# #
# #     # 模拟输入
# #     test_input = torch.randn(1, 32, 10, 31)
# #     test_y = torch.randn(1, 32, 10, 6)  # 原始6维标签
# #
# #     # 测试
# #     out = model(test_input)
# #     print("Forward pass successful.")
# #     model.training_step((test_input, test_y), 0)
# #     print("Training step with detailed logging successful.")
#
#
#
#
#
#
#
#
#
# from typing import Any, Dict, Tuple, Optional, List, Iterator
# from itertools import chain
# import torch
# import torch.nn as nn
# from lightning import LightningModule
# from torchmetrics import MeanMetric
# from src.models.components.mlp import MLP
# from src.models.components.weight_methods import WeightMethods
#
#
# class MultiTaskLSTMWithEncoderDecoder(nn.Module):
#     """结合编码器-解码器和多任务LSTM的模型"""
#
#     def __init__(
#             self,
#             input_size: int,
#             hidden_size: int,
#             task_sizes: list,  # 每个任务(分组)的输出大小
#             task_names: list = None,
#             # 编码器-解码器参数
#             encoder_dims: list = None,
#             decoder_dims: list = None,
#             # LSTM参数
#             dropout: float = 0.0,
#             shared_layers: int = 2,
#             task_specific_layers: int = 1,  # 允许为 0
#             # 编码器参数
#             encoder_batch_norm: bool = False,
#             # 解码器参数
#             decoder_batch_norm: bool = False,
#     ):
#         super().__init__()
#
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.task_sizes = task_sizes
#         self.num_tasks = len(task_sizes)
#         self.task_names = task_names or [f"task_{i}" for i in range(self.num_tasks)]
#         self.shared_layers = shared_layers
#         self.task_specific_layers = task_specific_layers
#
#         # 设置编码器维度
#         if encoder_dims is None:
#             encoder_dims = [input_size, 512, 512, 64]
#         self.encoder_dims = encoder_dims
#
#         # 设置解码器基础维度
#         if decoder_dims is None:
#             decoder_dims = [hidden_size, 512, 512]
#         else:
#             decoder_dims = [hidden_size] + decoder_dims[1:]
#
#         # 编码器
#         self.encoder = MLP(
#             channel_list=encoder_dims,
#             batch_norm=encoder_batch_norm,
#             dropout=dropout
#         )
#
#         # 共享LSTM
#         lstm_input_size = encoder_dims[-1]
#         self.shared_lstm = nn.LSTM(
#             lstm_input_size,
#             hidden_size,
#             shared_layers,
#             batch_first=True,
#             dropout=dropout if shared_layers > 1 else 0.0
#         )
#
#         # --- 【修改点1】任务特定LSTM (支持 0 层) ---
#         self.task_specific_lstms = nn.ModuleDict()
#         if self.task_specific_layers > 0:
#             for task_name in self.task_names:
#                 task_lstm = nn.LSTM(
#                     hidden_size,
#                     hidden_size,
#                     task_specific_layers,
#                     batch_first=True,
#                     dropout=dropout if task_specific_layers > 1 else 0.0
#                 )
#                 self.task_specific_lstms[task_name] = task_lstm
#
#         # 任务特定解码器
#         self.task_decoders = nn.ModuleDict()
#         for i, (task_name, task_size) in enumerate(zip(self.task_names, self.task_sizes)):
#             decoder_dims_with_output = decoder_dims + [task_size]
#             task_decoder = MLP(
#                 channel_list=decoder_dims_with_output,
#                 batch_norm=decoder_batch_norm,
#                 dropout=dropout
#             )
#             self.task_decoders[task_name] = task_decoder
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """前向传播"""
#         if x.dim() == 4:
#             x = x.squeeze(0) if x.size(0) == 1 else x
#
#         batch_size, seq_len, _ = x.shape
#
#         # Encoder
#         encoded_sequence = []
#         for t in range(seq_len):
#             encoded_sequence.append(self.encoder(x[:, t, :]).unsqueeze(1))
#         encoded_sequence = torch.cat(encoded_sequence, dim=1)
#
#         # Shared LSTM
#         # shared_output: [batch, seq_len, hidden_size]
#         shared_output, _ = self.shared_lstm(encoded_sequence)
#
#         # Task Specific Processing
#         task_outputs = []
#         for task_name in self.task_names:
#             # --- 【修改点2】如果存在特定LSTM，先通过它 ---
#             if self.task_specific_layers > 0:
#                 # task_lstm_output: [batch, seq_len, hidden_size]
#                 task_branch_output, _ = self.task_specific_lstms[task_name](shared_output)
#             else:
#                 # 否则直接使用共享层的输出
#                 task_branch_output = shared_output
#
#             # Decoder (逐时间步解码)
#             task_decoder = self.task_decoders[task_name]
#             task_time_outputs = []
#             for t in range(seq_len):
#                 # 输入是 [batch, hidden_size]
#                 step_input = task_branch_output[:, t, :]
#                 task_time_outputs.append(task_decoder(step_input).unsqueeze(1))
#
#             task_outputs.append(torch.cat(task_time_outputs, dim=1))
#
#         return torch.cat(task_outputs, dim=-1)
#
#     def get_task_output(self, x: torch.Tensor, task_index: int) -> torch.Tensor:
#         if task_index >= self.num_tasks:
#             raise ValueError(f"Task index {task_index} out of range")
#
#         if x.dim() == 4: x = x.squeeze(0)
#         batch_size, seq_len, _ = x.shape
#         task_name = self.task_names[task_index]
#
#         encoded_sequence = []
#         for t in range(seq_len):
#             encoded_sequence.append(self.encoder(x[:, t, :]).unsqueeze(1))
#         encoded_sequence = torch.cat(encoded_sequence, dim=1)
#
#         shared_output, _ = self.shared_lstm(encoded_sequence)
#
#         # --- 【修改点3】特定任务分支逻辑同步修改 ---
#         if self.task_specific_layers > 0:
#             task_branch_output, _ = self.task_specific_lstms[task_name](shared_output)
#         else:
#             task_branch_output = shared_output
#
#         task_decoder = self.task_decoders[task_name]
#         task_time_outputs = []
#         for t in range(seq_len):
#             task_time_outputs.append(task_decoder(task_branch_output[:, t, :]).unsqueeze(1))
#
#         return torch.cat(task_time_outputs, dim=1)
#
#     def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
#         """共享参数：编码器 + 共享LSTM"""
#         return chain(self.encoder.parameters(), self.shared_lstm.parameters())
#
#     def task_specific_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
#         """任务特定参数：特定LSTM(如果有) + 解码器"""
#         params = [decoder.parameters() for decoder in self.task_decoders.values()]
#         if self.task_specific_layers > 0:
#             params.append(self.task_specific_lstms.parameters())
#         return chain(*params)
#
#     def last_shared_parameters(self):
#         """最后共享层参数：LSTM最后一层"""
#         return list(self.shared_lstm.parameters())[-2:]
#
#
# class FtdModule(LightningModule):
#     def __init__(
#             self,
#             input_size: int,
#             hidden_size: int,
#             task_sizes: List[int],
#             task_names: List[str] = None,
#             task_indices: List[List[int]] = None,
#             original_task_names: List[str] = None,
#             encoder_dims: List[int] = None,
#             decoder_dims: List[int] = None,
#             shared_layers: int = 2,
#             task_specific_layers: int = 1,
#             dropout: float = 0.0,
#             learning_rate: float = 0.001,
#             loss_term: str = "",
#             encoder_batch_norm: bool = False,
#             decoder_batch_norm: bool = False,
#             optimizer: Optional[torch.optim.Optimizer] = None,
#             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
#             compile: bool = False,
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()
#
#         if task_indices is None:
#             current_idx = 0
#             self.task_indices = []
#             for size in task_sizes:
#                 self.task_indices.append(list(range(current_idx, current_idx + size)))
#                 current_idx += size
#         else:
#             self.task_indices = task_indices
#
#         self.original_task_names = original_task_names
#
#         self.net = MultiTaskLSTMWithEncoderDecoder(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             task_sizes=task_sizes,
#             task_names=task_names,
#             encoder_dims=encoder_dims,
#             decoder_dims=decoder_dims,
#             dropout=dropout,
#             shared_layers=shared_layers,
#             task_specific_layers=task_specific_layers,
#             encoder_batch_norm=encoder_batch_norm,
#             decoder_batch_norm=decoder_batch_norm,
#         )
#
#         self.criterion = torch.nn.MSELoss(reduction="none")
#
#         # 指标定义
#         self.train_loss = MeanMetric()
#         self.val_loss = MeanMetric()
#         self.test_loss = MeanMetric()
#
#         self.train_group_losses = nn.ModuleList([MeanMetric() for _ in range(len(task_sizes))])
#         self.val_group_losses = nn.ModuleList([MeanMetric() for _ in range(len(task_sizes))])
#         self.test_group_losses = nn.ModuleList([MeanMetric() for _ in range(len(task_sizes))])
#
#         if self.original_task_names:
#             self.train_detail_losses = nn.ModuleDict({name: MeanMetric() for name in self.original_task_names})
#             self.val_detail_losses = nn.ModuleDict({name: MeanMetric() for name in self.original_task_names})
#             self.test_detail_losses = nn.ModuleDict({name: MeanMetric() for name in self.original_task_names})
#         else:
#             self.train_detail_losses = nn.ModuleDict()
#             self.val_detail_losses = nn.ModuleDict()
#             self.test_detail_losses = nn.ModuleDict()
#
#         self.saved_x = None;
#         self.saved_y = None;
#         self.saved_preds = None
#         self.loss_term = loss_term
#         self.task_num = len(task_sizes)
#         self.automatic_optimization = False
#
#     def setup(self, stage: str) -> None:
#         if self.hparams.compile and stage == "fit":
#             self.net = torch.compile(self.net)
#         if stage == "fit" and self.loss_term in ["stch", "fairgrad", "famo"]:
#             self.weight_method = WeightMethods(self.loss_term, self.task_num, self.device)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)
#
#     def compute_task_losses(self, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """计算分组损失 (用于优化)"""
#         task_losses = []
#         pred_start_idx = 0
#         for i, task_size in enumerate(self.hparams.task_sizes):
#             pred_end_idx = pred_start_idx + task_size
#             task_pred = preds[:, :, pred_start_idx:pred_end_idx]
#             task_target = y[:, :, self.task_indices[i]]
#             task_losses.append(self.criterion(task_pred, task_target).mean())
#             pred_start_idx = pred_end_idx
#         return torch.stack(task_losses)
#
#     def compute_and_log_detailed_losses(self, preds: torch.Tensor, y: torch.Tensor, stage: str):
#         """计算并记录详细的单分量损失"""
#         if not self.original_task_names:
#             return
#
#         metrics_dict = getattr(self, f"{stage}_detail_losses")
#         pred_ptr = 0
#
#         for group_idx, indices_in_group in enumerate(self.task_indices):
#             for i, original_idx in enumerate(indices_in_group):
#                 pred_slice = preds[:, :, pred_ptr + i: pred_ptr + i + 1]
#                 target_slice = y[:, :, original_idx: original_idx + 1]
#                 loss = self.criterion(pred_slice, target_slice).mean()
#
#                 if original_idx < len(self.original_task_names):
#                     name = self.original_task_names[original_idx]
#                     if name in metrics_dict:
#                         metrics_dict[name](loss)
#                         self.log(f"{stage}/detail_{name}_loss", metrics_dict[name], on_step=False, on_epoch=True,
#                                  prog_bar=False)
#
#             pred_ptr += len(indices_in_group)
#
#     def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
#         x, y = batch
#         x = x.squeeze(0) if x.dim() == 4 else x
#         y = y.squeeze(0) if y.dim() == 4 else y
#
#         preds = self.forward(x)
#         opt = self.optimizers()
#         opt.zero_grad()
#
#         group_losses = self.compute_task_losses(preds, y)
#
#         if self.loss_term in ["stch", "fairgrad", "famo"]:
#             _, extra = self.weight_method.backward(
#                 losses=group_losses,
#                 shared_parameters=list(self.net.shared_parameters()),
#                 task_specific_parameters=list(self.net.task_specific_parameters()),
#                 last_shared_parameters=list(self.net.last_shared_parameters()),
#                 epoch=self.current_epoch,
#             )
#             total_loss = group_losses.mean()
#         else:
#             total_loss = group_losses.sum()
#             self.manual_backward(total_loss)
#
#         opt.step()
#
#         if "famo" in self.loss_term:
#             with torch.no_grad():
#                 preds_new = self.forward(x)
#                 group_losses_new = self.compute_task_losses(preds_new, y)
#                 self.weight_method.method.update(group_losses_new.detach())
#
#         self.train_loss(total_loss)
#         self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
#
#         for i, loss in enumerate(group_losses):
#             name = self.net.task_names[i]
#             self.train_group_losses[i](loss)
#             self.log(f"train/group_{name}_loss", self.train_group_losses[i], on_step=True, on_epoch=True,
#                      prog_bar=False)
#
#         with torch.no_grad():
#             self.compute_and_log_detailed_losses(preds, y, "train")
#
#         return total_loss
#
#     def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
#         x, y = batch
#         x = x.squeeze(0) if x.dim() == 4 else x
#         y = y.squeeze(0) if y.dim() == 4 else y
#
#         preds = self.forward(x)
#         group_losses = self.compute_task_losses(preds, y)
#         total_loss = group_losses.mean()
#
#         self.val_loss(total_loss)
#         self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
#
#         for i, loss in enumerate(group_losses):
#             name = self.net.task_names[i]
#             self.val_group_losses[i](loss)
#             self.log(f"val/group_{name}_loss", self.val_group_losses[i], on_step=False, on_epoch=True, prog_bar=True)
#
#         self.compute_and_log_detailed_losses(preds, y, "val")
#
#     def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
#         x, y = batch
#         x = x.squeeze(0) if x.dim() == 4 else x
#         y = y.squeeze(0) if y.dim() == 4 else y
#
#         preds = self.forward(x)
#         group_losses = self.compute_task_losses(preds, y)
#         total_loss = group_losses.mean()
#
#         self.test_loss(total_loss)
#         self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
#
#         for i, loss in enumerate(group_losses):
#             name = self.net.task_names[i]
#             self.test_group_losses[i](loss)
#             self.log(f"test/group_{name}_loss", self.test_group_losses[i], on_step=False, on_epoch=True, prog_bar=True)
#
#         self.compute_and_log_detailed_losses(preds, y, "test")
#
#         self.saved_x = x.detach();
#         self.saved_y = y.detach();
#         self.saved_preds = preds.detach()
#
#     def on_test_epoch_end(self) -> None:
#         if self.saved_x is not None:
#             torch.save(self.saved_x, "saved_x.pth")
#             torch.save(self.saved_y, "saved_y.pth")
#             torch.save(self.saved_preds, "saved_preds.pth")
#
#     def configure_optimizers(self) -> Dict[str, Any]:
#         if hasattr(self, 'hparams') and hasattr(self.hparams, 'optimizer') and self.hparams.optimizer is not None:
#             if isinstance(self.hparams.optimizer, dict):
#                 optimizer_class = self.hparams.optimizer.get('_target_', torch.optim.Adam)
#                 optimizer_params = {k: v for k, v in self.hparams.optimizer.items() if k != '_target_'}
#                 optimizer_params['params'] = self.parameters()
#                 optimizer = optimizer_class(**optimizer_params)
#             elif callable(self.hparams.optimizer):
#                 optimizer = self.hparams.optimizer(params=self.parameters())
#             else:
#                 lr = getattr(self, 'learning_rate', getattr(self.hparams, 'learning_rate', 0.001))
#                 optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#         else:
#             lr = getattr(self, 'learning_rate', getattr(self.hparams, 'learning_rate', 0.001))
#             optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#
#         if hasattr(self, 'hparams') and hasattr(self.hparams, 'scheduler') and self.hparams.scheduler is not None:
#             scheduler = None
#             if isinstance(self.hparams.scheduler, dict):
#                 scheduler_class = self.hparams.scheduler.get('_target_', None)
#                 if scheduler_class is not None:
#                     valid_params = ['optimizer', 'mode', 'factor', 'patience', 'verbose', 'threshold', 'threshold_mode',
#                                     'cooldown', 'min_lr', 'eps']
#                     scheduler_params = {k: v for k, v in self.hparams.scheduler.items() if
#                                         k != '_target_' and k in valid_params}
#                     scheduler_params['optimizer'] = optimizer
#                     try:
#                         scheduler = scheduler_class(**scheduler_params)
#                     except TypeError:
#                         scheduler = None
#             if scheduler is not None:
#                 return {"optimizer": optimizer,
#                         "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss", "interval": "epoch",
#                                          "frequency": 1}}
#         return {"optimizer": optimizer}