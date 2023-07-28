# Everything

## Description

为简化测试，从 FSRCNN 中抽取了一层分布较为复杂的卷积层进行数据提取，本层结构可使用以下简化网络进行还原：

```Python
class HardwareModel(nn.Module):
    def __init__(self):
        super(HardwareModel, self).__init__()
        self.conv1 = nn.Conv2d(12, 12, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return x
```

由于测试过程中 bias 的加入需要自定义量化进行，因此不允许网络内部自动加入，因此设置为 False，但 bias 参数以提取，后续可手动加入。

本次共从网络中提取了 `input`，`output`，`weight`，`bias` 四项数据，每项数据都有 `original`，`int` 和 `quantized`  三个版本，可任意使用。此外，`input` 和 `output` 还划分了原始数据版本和 `patch` 版本。

使用数据时可直接使用：

```Python
everything = torch.load('everything.pth')
```

加载数据，并根据下一章节的数据格式进行使用。

## Data Format

- everything (Dict)
  - weight (Dict)
    - weight_original (Tensor)
    - weight_int (Tensor)
    - weight_quantized (Tensor)
    - weight_parameter_dict (Dict)
      - weight_original_min
      - weight_original_max
      - weight_int_min
      - weight_int_max
      - weight_quantized_min
      - weight_quantized_max
      - weight_scale
      - weight_zero = 0
      - weight_width
  - bias (Dict)
    - bias_original (Tensor)
    - bias_int (Tensor)
    - bias_quantized (Tensor)
    - bias_parameter_dict (Dict)
      - bias_original_min
      - bias_original_max
      - bias_int_min
      - bias_int_max
      - bias_quantized_min
      - bias_quantized_max
      - bias_scale
      - bias_zero = 0
      - bias_width
  - input (Dict)
    - input_original (Tensor)
    - input_int (Tensor)
    - input_quantized (Tensor)
    - input_original_patch_list (List)
      - input_original_patch (Tensor)
    - input_int_patch_list (List)
      - input_int_patch (Tensor)
    - input_quantized_patch_list (List)
      - input_quantized_patch (Tensor)
    - input_parameter_dict (Dict)
      - input_original_min
      - input_original_max
      - input_int_min
      - input_int_max
      - input_quantized_min
      - input_quantized_max
      - input_scale
      - input_zero = 0
      - input_width
  - output (Dict)
    - output_original (Tensor)
    - output_int (Tensor)
    - output_quantized (Tensor)
    - output_original_patch_list (List)
      - output_original_patch (Dict)
        - reg_1 (List)
          - (1), (2), (3), (4)
        - reg_2 (List)
          - (1 + 2), (3 + 4)
        - reg_3 (Tensor)
          - (1 + 2 + 3 + 4)
    - output_int_patch_list (List)
      - output_int_patch (Dict)
        - reg_1 (List)
          - (1), (2), (3), (4)
        - reg_2 (List)
          - (1 + 2), (3 + 4)
        - reg_3 (Tensor)
          - (1 + 2 + 3 + 4)
    - output_quantized_patch_list (List)
      - output_quantized_patch (Dict)
        - reg_1 (List)
          - (1), (2), (3), (4)
        - reg_2 (List)
          - (1 + 2), (3 + 4)
        - reg_3 (Tensor)
          - (1 + 2 + 3 + 4)
    - output_parameter_dict (Dict)
      - output_original_min
      - output_original_max
      - output_int_min
      - output_int_max
      - output_quantized_min
      - output_quantized_max
      - output_scale = input_scale * weight_scale
      - output_zero = 0
      - output_width (Dict)
        - reg_1
        - reg_2 = reg_1 + 1
        - reg_3 = reg_2 + 1

## Data Transformation

$\mathrm{xxx\_quantized} = \mathrm{xxx\_int} \times \mathrm{xxx\_scale}$

## Dimension Information

`weight`, `bias` 无需多做介绍，就是常规的数据维度。

`input` 和 `output` 中，名称不含 `patch` 的部分都是正常数据维度排布，只有分了 patch 的部分需要大致说明。

`input_xxx_patch_list` 中，每个元素都是一个分好的单独的 patch，维度是 $4\times 4k \times 32 \times 32$，是从原始的 $1\times 4k \times 32 \times 32$ 转换而来，将第二个维度的 $4k$ 四等分，每一份中只保留一个 $k$ 的部分，剩下 $3k$ 用 0 填充，保留每一个 $k$ 的 Tensor 被放置在了 PyTorch 数据格式的 batch 维度，也就是第一个维度。填 0 操作保证了转换后的数据通道数和原始数据所使用的网络所对应，不必修改网络即可进行测试，在推理中，每四分之一的 channel 都可在 batch 维度独立进行，且实际都只使用了四分之一的卷积核输入通道，待输出后相加整合即可。综上，`input_xxx_patch_list[index][0]` 可取出 patch 的第一个部分，是一个三维 Tensor，其中 `[0: k]` 是有效的，其余为 0；`input_xxx_patch_list[index][1]` 可取出 patch 的第二个部分，是一个三维 Tensor，其中 `[k: 2k]` 是有效的，其余为 0；`input_xxx_patch_list[index][2]` 可取出 patch 的第三个部分，是一个三维 Tensor，其中 `[2k: 3k]` 是有效的，其余为 0；`input_xxx_patch_list[index][3]` 可取出 patch 的第四个部分，是一个三维 Tensor，其中 `[3k: 4k]` 是有效的，其余为 0。

`output_xxx_patch_list` 中，每个元素都是一个分好的单独的 patch，维度是 $4\times 4n \times 32 \times 32$，以第一个维度为标准进行求和即可得到 $1\times 4n \times 32 \times 32$ 的原始输出。