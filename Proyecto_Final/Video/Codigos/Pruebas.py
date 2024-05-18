import numpy as np
from mnist_loader import load_mnist
import Dense

net = Dense.load_pretrained_net("Codigos/wb.txt")

print(net.layers)

print(round(255/268, 2))

