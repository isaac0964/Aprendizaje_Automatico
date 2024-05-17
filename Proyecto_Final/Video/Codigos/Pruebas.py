import numpy as np
from mnist_loader import load_mnist
import Dense

net = Dense.load_pretrained_net("Codigos/wb.txt")

print(net.layers)

print(19.5//5)
print(19.5%5)
