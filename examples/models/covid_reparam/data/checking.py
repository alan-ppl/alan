import torch as t
import math
from torch.distributions import NegativeBinomial

# dist = NegativeBinomial(total_count = 1000, logits=(1000/(1000 + 100 * math.exp(-0.02))))

# print(dist.mean)

# dist = NegativeBinomial(total_count = 1000, logits=(1000/(1000 + 100)))


# print(dist.mean)

# dist = NegativeBinomial(total_count = 1000, probs=1-(1000/(1000 + 100 * math.exp(-0.02))))

# print(dist.mean)

# dist = NegativeBinomial(total_count = 1000, probs=1-(1000/(1000 + 100)))

# print(dist.mean)


# dist = NegativeBinomial(total_count = 1000, probs=(1000/(1000 + 100 * math.exp(0.02))))

# print(dist.mean)

# dist = NegativeBinomial(total_count = 1000, probs=(1000/(1000 + 100)))

# print(dist.mean)

ActiveCMs_NPIs = t.load('ActiveCMs_NPIs.pt')
print(ActiveCMs_NPIs.shape)

c4_3plus = ActiveCMs_NPIs[:, 4]
c4_2plus = ActiveCMs_NPIs[:, 7]
c4_full = ActiveCMs_NPIs[:, 8]

print((c4_3plus == c4_2plus))
print((c4_3plus == c4_full))