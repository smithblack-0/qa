from Model import embedding
import torch

hashembed = embedding.HashEmbedding(10, 64, 10, 4)

tensor = torch.randint(0, 20, [10, 64])
print(tensor.shape)
print(hashembed(tensor))