from typing import List, Union, Tuple, Optional
import torch
import torch.nn as nn

Device = Union[None, str, torch.device]


# noinspection PyAbstractClass
class MLP(nn.Module):
    __slots__ = 'layers', 'in_features', 'out_features'

    in_features: int
    out_features: int
    layers: nn.ModuleList

    def __init__(self,
                 layer_sizes: List[int],
                 in_features: int,
                 activation: nn.Module = None,
                 out_features: Union[None, int, Tuple[int, nn.Module]] = None):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        self.in_features = in_features
        self.layers = nn.ModuleList()
        prev_features = in_features
        for size in layer_sizes:
            layer = nn.Linear(prev_features, size)
            self.layers.append(layer)
            self.layers.append(activation)
            prev_features = size

        if out_features is not None:
            if isinstance(out_features, int):
                self.layers.append(nn.Linear(prev_features, out_features))
                prev_features = out_features
            else:
                self.layers.append(nn.Linear(prev_features, out_features[0]))
                self.layers.append(out_features[1])
                prev_features, _ = prev_features
        self.out_features = prev_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# noinspection PyAbstractClass
class LSTM(nn.Module):
    __slots__ = 'layers', 'in_features', 'out_features'

    in_features: int
    out_features: int
    layers: nn.ModuleList

    def __init__(self,
                 layer_sizes: List[int],
                 in_features: int,
                 activation: nn.Module = None,
                 out_features: Union[None, int, Tuple[int, nn.Module]] = None):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        self.in_features = in_features
        self.layers = nn.ModuleList()
        prev_features = in_features
        for size in layer_sizes:
            layer = nn.LSTMCell(prev_features, size)
            self.layers.append(layer)
            self.layers.append(activation)
            prev_features = size

        if out_features is not None:
            if isinstance(out_features, int):
                self.layers.append(nn.LSTMCell(prev_features, out_features))
                prev_features = out_features
            else:
                self.layers.append(nn.LSTMCell(prev_features, out_features[0]))
                self.layers.append(out_features[1])
                prev_features, _ = prev_features
        self.out_features = prev_features

    def mem_size(self) -> Tuple[int, ...]:
        return tuple(layer.hidden_size for layer in self.layers
                     if isinstance(layer, nn.LSTMCell) for _ in range(2))

    def init_mem(self,
                 batch_size: int,
                 device: Device = None) -> Tuple[torch.Tensor, ...]:
        mem = []
        for layer in self.layers:
            if isinstance(layer, nn.LSTMCell):
                mem.append(
                    torch.zeros((batch_size, layer.hidden_size)
                                if batch_size else layer.hidden_size,
                                dtype=torch.float32,
                                device=device))
                mem.append(
                    torch.zeros((batch_size, layer.hidden_size)
                                if batch_size else layer.hidden_size,
                                dtype=torch.float32,
                                device=device))
        return tuple(mem)

    def forward(
        self, x: torch.Tensor, mem: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        mem_1 = []
        i = 0
        for layer in self.layers:
            if isinstance(layer, nn.LSTMCell):
                x, c = layer(x, (mem[i], mem[i + 1]))
                mem_1.append(x)  # hidden state
                mem_1.append(c)  # cell state
                i += 2
            else:
                x = layer(x)
        return x, tuple(mem_1)


# noinspection PyAbstractClass
class ConvNet(nn.Module):
    __slots__ = 'layers',

    layers: nn.ModuleList

    def __init__(self,
                 layers: List[Union[Tuple[int, int, int], Tuple[int, int]]],
                 in_channels: int,
                 flatten: bool = False):
        super().__init__()
        prev_channels = in_channels
        self.layers = nn.ModuleList()
        for layer in layers:
            if len(layer) == 3:
                channels, kernel, stride = layer
            else:
                channels, kernel = layer
                stride = 1
            layer = nn.Conv2d(prev_channels, channels, kernel, stride)
            self.layers.append(layer)

            self.layers.append(nn.ReLU())
            prev_channels = channels
        if flatten:
            self.layers.append(nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# noinspection PyAbstractClass
class DuelingQNet(nn.Module):
    __slots__ = 'features_net', 'memory_net', 'value_net', 'advantage_net'

    features_net: nn.Module
    memory_net: Optional[nn.Module]
    value_net: nn.Module
    advantage_net: nn.Module

    def __init__(self,
                 *,
                 feature_net: nn.Module,
                 value_net: nn.Module,
                 advantage_net: nn.Module,
                 memory_net: Optional[nn.Module] = None):
        super().__init__()
        self.features_net = feature_net
        self.memory_net = memory_net
        self.value_net = value_net
        self.advantage_net = advantage_net

    def mem_size(self) -> Tuple[int, ...]:
        return self.memory_net.mem_size()

    def init_mem(self,
                 batch_size: int,
                 device: Device = None) -> Tuple[torch.Tensor, ...]:
        return self.memory_net.init_mem(batch_size, device)

    def remember(self, x: torch.Tensor,
                 mem: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x = self.features_net(x)
        _, mem = self.memory_net(x, mem)
        return mem

    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        x = self.features_net(x)
        if self.memory_net is not None:
            x, mem = self.memory_net(x, mem)
        advantages: torch.Tensor = self.advantage_net(x)
        qs: torch.Tensor = self.value_net(x) + advantages - advantages.mean(
            1, True)
        if mem is not None:
            return qs, mem
        else:
            return qs


# noinspection PyAbstractClass
class QuantileDuelingQNet(DuelingQNet):
    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        features: torch.Tensor = self.features_net(x)
        if self.memory_net is not None:
            features, mem = self.memory_net(features, mem)
        values: torch.Tensor = self.value_net(features).unsqueeze(
            1)  # -1, 1, Q
        num_quantiles: int = values.size()[2]
        advantages: torch.Tensor = self.advantage_net(features)  # -1, A*Q
        advantages = advantages.view(-1,
                                     advantages.size()[1] // num_quantiles,
                                     num_quantiles)  # -1, A, Q
        qs = values + advantages - advantages.mean(1, True)  # -1, A, Q
        if mem is not None:
            return qs, mem
        else:
            return qs
