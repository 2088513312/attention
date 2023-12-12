import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

data = pd.read_csv("LLA.csv",header=None)
data=data.dropna()
class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionModel, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, q,k,v):

        # attention_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5), dim=2)
        attention_weights = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)  #【180，4】【4，180】 [180,180]
        out = torch.bmm(attention_weights, v)

        out = out/400

        return out, attention_weights


model = AttentionModel(input_size=180, hidden_size=64)

tensor_data= torch.tensor(data.to_numpy())
out = tensor_data.unsqueeze(0)
out, attention_weights = model(out, out, out)
Vis_data = tensor_data.squeeze(0)
Attn_data = out.squeeze(0)
attention = attention_weights.squeeze(0)
m = (attention-torch.min(attention))/(torch.max(attention)-torch.min(attention))
print(attention)
print(m)
print(len(attention))
plt.imshow(m, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(0,181,20))
plt.show()


fig, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
for i in range(4):
    ax = axes[i]
    # ax.spines['top'].set_visible(False)
    ax.plot(Vis_data[:, i]*5, color='blue', linewidth=1)
    ax.set_title(f"Lead {i + 1}")
    y_bottom = ax.get_ylim()[0]
    ax.fill_between(np.arange(180), Attn_data[:, i],interpolate=True, color='red', alpha=0.3)
    ax.set_xticks(range(0,181,10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, int(max(Attn_data[:, i])))
    ax.set_xlim(0, 180)
plt.tight_layout()
# plt.savefig('ECG_Attention.png', dpi=300, format='png')
plt.show()


