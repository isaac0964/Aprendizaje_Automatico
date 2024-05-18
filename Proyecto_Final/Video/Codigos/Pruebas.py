import numpy as np
from mnist_loader import load_mnist
import Dense

net = Dense.load_pretrained_net("Codigos/wb.txt")

print(net.layers)

for i in range(4):
    print(i*2)
