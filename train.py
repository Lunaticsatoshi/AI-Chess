import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import optim

class ChessDataset(Dataset):
    def __init__(self):
        data = np.load("processed/dataset_250.npz")
        self.X = data['arr_0']
        self.Y = data['arr_1']
        print("loaded", self.X.shape, self.Y.shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        #First Layer
        self.a1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        #second Layer
        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        #Third Layer
        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        #Fourth Layer
        self.d1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.d3 = nn.Conv2d(128, 256, kernel_size=1, stride=2)

        #Fifth Layer
        self.e1 = nn.Conv2d(256, 256, kernel_size=2, padding=1)
        self.e2 = nn.Conv2d(256, 256, kernel_size=2, padding=1)
        self.e3 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        #Sixth Layer
        self.f1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.f2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.f3 = nn.Conv2d(512, 256, kernel_size=1, stride=2)

        #Seventh Layer
        self.g1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.g2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.g3 = nn.Conv2d(256, 256, kernel_size=1, stride=2)

        #Eight Layer
        self.h1 = nn.Conv2d(256, 256, kernel_size=2, padding=1)
        self.h2 = nn.Conv2d(256, 256, kernel_size=2, padding=1)
        self.h3 = nn.Conv2d(256, 128, kernel_size=2, stride=2)

        #Ninth Layer
        self.i1 = nn.Conv2d(128, 128, kernel_size=1)
        self.i2 = nn.Conv2d(128, 128, kernel_size=1)
        self.i3 = nn.Conv2d(128, 128, kernel_size=1)

        #Final Layer
        self.last = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.a1(x))
        x = torch.relu(self.a2(x))
        x = torch.relu(self.a3(x))

        x = torch.relu(self.b1(x))
        x = torch.relu(self.b2(x))
        x = torch.relu(self.b3(x))

        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))

        x = torch.relu(self.d1(x))
        x = torch.relu(self.d2(x))
        x = torch.relu(self.d3(x))

        x = torch.relu(self.e1(x))
        x = torch.relu(self.e2(x))
        x = torch.relu(self.e3(x))

        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        x = torch.relu(self.f3(x))

        x = torch.relu(self.g1(x))
        x = torch.relu(self.g2(x))
        x = torch.relu(self.g3(x))

        x = torch.relu(self.h1(x))
        x = torch.relu(self.h2(x))
        x = torch.relu(self.h3(x))

        x = torch.relu(self.i1(x))
        x = torch.relu(self.i2(x))
        x = torch.relu(self.i3(x))

        x = x.view(-1, 128)
        x = self.last(x)
        
        #value Of Output
        return torch.tanh(x)

if __name__ == "__main__":
    device = "cpu"
    chess_dataset = ChessDataset()
    train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=256, shuffle=True)
    model = NeuralNet()
    optimizer = optim.Adam(model.parameters())
    floss = nn.MSELoss()

    if device == "cuda":
        model.cuda()
        
    model.train()

    for epoch in range(1):
        all_loss = 0
        num_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.unsqueeze(-1)
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()

            #print(data.shape, target.shape)
            optimizer.zero_grad()
            output = model(data)
            #print(output.shape)

            loss = floss(output, target)
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            num_loss += 1

        print("%3d: %f" %(epoch, all_loss/num_loss))
        torch.save(model.state_dict(), "model/modelbig.pth")