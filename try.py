# import torch.nn as nn
# import torch

# layer = eval("nn.ReLU")
# print(layer)

# a = torch.Tensor(1)

# print(layer(a))

# import pandas as pd

# # df = pd.read_csv("./results_ade.csv", index_col=0)
# df = pd.DataFrame(
#     {
#         "name":  1,
#         "eth":   1,
#         "hotel": 2,
#         "univ":  2,
#         "zara1": 3,
#         "zara2": 4,
#         "avg":   5
#     }, index=[0]
# )
# # df = pd.concat([df, df_a])

# df.to_csv("./results_ade.csv", index=False)

# # df = pd.read_csv("./results_fde.csv", index_col=0)
# # df_a = pd.DataFrame(
# #     {
# #         "name":  1,
# #         "eth":   1,
# #         "hotel": 2,
# #         "univ":  2,
# #         "zara1": 3,
# #         "zara2": 4,
# #         "avg":   5
# #     }
# # )
# # df = pd.concat([df, df_a])

# df.to_csv("./results_fde.csv", index=False)
import torch
import math
import torch.nn as nn
from einops import rearrange


class T5RelativePositionBias(nn.Module):

    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  causal=True,
                                  num_buckets=32,
                                  max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) /
                                    math.log(max_distance / max_exact) *
                                    (num_buckets - max_exact)).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale