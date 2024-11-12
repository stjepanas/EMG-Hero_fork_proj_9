import dataclasses

import torch
import torch.nn as nn
import d3rlpy


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_batch_norm=False, dropout=0.1, output_activation=None):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.Dropout(p=dropout))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU(inplace=True))

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.Dropout(p=dropout))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, x, temperature=1.0):
        x = self.network(x)
        if self.output_activation is not None:
            x = x / temperature
            x = self.output_activation(x)
        return x



class Conv1DEncoder(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super().__init__()
        self.feature_size = feature_size
        input_size = observation_shape[0]
        if action_size is not None:
            input_size += action_size

        # self.fc1 = nn.Linear(input_size, feature_size)

        in_channels = 1
        out_channels = 1
        kernel_size_in = 3
        kernel_size_1 = 6
        stride = 1
        padding = 1

        self.conv_in = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size_in,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size_1,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.0, inplace=False)

        # self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        # self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size_2,
        #                        stride=stride, padding=padding, bias=False)

        self.fc_out = nn.Linear(feature_size, feature_size)

    def forward(self, x, action):
        h = torch.cat([x, action], dim=1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return h


@dataclasses.dataclass()
class Conv1DEncoderFactory(d3rlpy.models.EncoderFactory):
    feature_size: int

    def create(self, observation_shape):
        action_size = None
        return Conv1DEncoder(observation_shape, action_size, self.feature_size)

    def create_with_action(self, observation_shape, action_size, discrete_action):
        return Conv1DEncoder(observation_shape, action_size, self.feature_size)

    @staticmethod
    def get_type() -> str:
        return "custom"