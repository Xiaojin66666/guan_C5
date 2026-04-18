# from typing import Iterator
# from itertools import chain
# import torch
# import torch.nn as nn
# from src.models.components.mlp import MLP
#
#
# class MultiTaskLSTMWithEncoderDecoder(nn.Module):
#     """结合编码器-解码器和多任务LSTM的模型"""
#
#     def __init__(
#             self,
#             input_size: int,
#             hidden_size: int,
#             num_layers: int,
#             task_sizes: list,  # 每个任务的输出大小
#             task_names: list = None,
#             # 编码器-解码器参数
#             encoder_dims: list = None,  # 编码器维度 [input_size, hidden1, hidden2, ...]
#             decoder_dims: list = None,  # 解码器基础维度，每个任务会单独创建解码器
#             # LSTM参数
#             dropout: float = 0.0,
#             shared_layers: int = 2,  # 共享LSTM层数
#             task_specific_layers: int = 1,  # 任务特定层数
#             # 编码器参数
#             encoder_batch_norm: bool = False,
#             # 解码器参数
#             decoder_batch_norm: bool = False,
#     ):
#         super().__init__()
#
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
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
#             decoder_dims = [64, 512, 512]  # 不包含输出层，输出层由task_sizes决定
#
#         # 编码器：将输入映射到潜在空间
#         self.encoder = MLP(
#             channel_list=encoder_dims,
#             batch_norm=encoder_batch_norm,
#             dropout=dropout
#         )
#
#         # 共享的LSTM层 - 处理编码后的序列
#         lstm_input_size = encoder_dims[-1]  # 编码器输出维度作为LSTM输入
#         self.shared_lstm = nn.LSTM(
#             lstm_input_size,
#             hidden_size,
#             shared_layers,
#             batch_first=True,
#             dropout=dropout if shared_layers > 1 else 0.0
#         )
#
#         # 任务特定的解码器
#         self.task_decoders = nn.ModuleDict()
#         for i, (task_name, task_size) in enumerate(zip(self.task_names, self.task_sizes)):
#             # 每个任务有自己的解码器
#             decoder_dims_with_output = decoder_dims + [task_size]
#             task_decoder = MLP(
#                 channel_list=decoder_dims_with_output,
#                 batch_norm=decoder_batch_norm,
#                 dropout=dropout
#             )
#             self.task_decoders[task_name] = task_decoder
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """前向传播
#
#         Args:
#             x: 输入序列 [batch, seq_len, input_size] 或 [1, batch, seq_len, input_size]
#
#         Returns:
#             所有任务的输出拼接 [batch, seq_len, sum(task_sizes)]
#         """
#         # 处理4D输入
#         if x.dim() == 4:
#             if x.size(0) == 1:
#                 x = x.squeeze(0)
#             else:
#                 raise ValueError(f"不支持的输入形状: {x.shape}")
#
#         batch_size, seq_len, input_dim = x.shape
#
#         # 编码器处理每个时间步
#         encoded_sequence = []
#         for t in range(seq_len):
#             time_step = x[:, t, :]  # [batch, input_size]
#             encoded_time_step = self.encoder(time_step)  # [batch, encoder_output_size]
#             encoded_sequence.append(encoded_time_step.unsqueeze(1))
#
#         encoded_sequence = torch.cat(encoded_sequence, dim=1)  # [batch, seq_len, encoder_output_size]
#
#         # 共享LSTM处理编码后的序列
#         lstm_output, (hidden, cell) = self.shared_lstm(encoded_sequence)
#
#         # 各任务解码器处理
#         task_outputs = []
#         for task_name in self.task_names:
#             task_decoder = self.task_decoders[task_name]
#
#             # 对每个时间步应用任务解码器
#             task_time_outputs = []
#             for t in range(seq_len):
#                 time_step_lstm_output = lstm_output[:, t, :]  # [batch, hidden_size]
#                 task_time_output = task_decoder(time_step_lstm_output)  # [batch, task_size]
#                 task_time_outputs.append(task_time_output.unsqueeze(1))
#
#             task_output = torch.cat(task_time_outputs, dim=1)  # [batch, seq_len, task_size]
#             task_outputs.append(task_output)
#
#         # 拼接所有任务的输出
#         return torch.cat(task_outputs, dim=-1)
#
#     def get_task_output(self, x: torch.Tensor, task_index: int) -> torch.Tensor:
#         """获取特定任务的输出"""
#         if task_index >= self.num_tasks:
#             raise ValueError(f"任务索引 {task_index} 超出范围，共有 {self.num_tasks} 个任务")
#
#         # 处理输入
#         if x.dim() == 4:
#             x = x.squeeze(0)
#
#         batch_size, seq_len, input_dim = x.shape
#         task_name = self.task_names[task_index]
#
#         # 编码器处理
#         encoded_sequence = []
#         for t in range(seq_len):
#             time_step = x[:, t, :]
#             encoded_time_step = self.encoder(time_step)
#             encoded_sequence.append(encoded_time_step.unsqueeze(1))
#         encoded_sequence = torch.cat(encoded_sequence, dim=1)
#
#         # LSTM处理
#         lstm_output, (hidden, cell) = self.shared_lstm(encoded_sequence)
#
#         # 特定任务解码
#         task_decoder = self.task_decoders[task_name]
#         task_time_outputs = []
#         for t in range(seq_len):
#             time_step_lstm_output = lstm_output[:, t, :]
#             task_time_output = task_decoder(time_step_lstm_output)
#             task_time_outputs.append(task_time_output.unsqueeze(1))
#
#         return torch.cat(task_time_outputs, dim=1)
#
#     def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
#         """共享参数：编码器和共享LSTM"""
#         return chain(
#             self.encoder.parameters(),
#             self.shared_lstm.parameters()
#         )
#
#     def task_specific_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
#         """任务特定参数：各任务的解码器"""
#         return chain(*[decoder.parameters() for decoder in self.task_decoders.values()])
#
#     def last_shared_parameters(self):
#         """最后共享层参数：LSTM最后一层"""
#         return list(self.shared_lstm.parameters())[-2:]  # 最后一层的权重和偏置