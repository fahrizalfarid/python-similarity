# python-similarity
```python
import torch, numpy as np, pandas as pd

device = "cuda"
batch_size = 10000
features = 3

model = torch.jit.load("similarity.pt").to(device)
model_norm = torch.jit.load("similarity-normalize.pt").to(device)

a = torch.tensor(
    np.random.uniform(low=10, high=25, size=(batch_size, features)).astype(np.float32)
).to(device)
b = torch.tensor(
    np.random.uniform(low=10, high=25, size=(batch_size, features)).astype(np.float32)
).to(device)

with torch.no_grad():
    y = model(a, b)
    a_norm, b_norm, y_norm = model_norm(a, b)

# # cpu
# df = pd.DataFrame(
#     {
#         "a": a.numpy().tolist(),
#         "a_norm": a_norm.numpy().tolist(),
#         "b": b.numpy().tolist(),
#         "b_norm": b_norm.numpy().tolist(),
#         "result": y.numpy(),
#         "result-norm": y_norm.numpy(),
#     }
# )

# gpu
df = pd.DataFrame(
    {
        "a": a.detach().cpu().numpy().tolist(),
        "a_norm": a_norm.detach().cpu().numpy().tolist(),
        "b": b.detach().cpu().numpy().tolist(),
        "b_norm": b_norm.detach().cpu().numpy().tolist(),
        "result": y.detach().cpu().numpy(),
        "result-norm": y_norm.detach().cpu().numpy(),
    }
)

df.to_csv(
    "result.csv",
    sep=",",
    index=False,
)
print("done")
```
