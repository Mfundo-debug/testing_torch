from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
datasets = DataLoader(train_data, batch_size=64, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork()

learning_rate = 1e-3
optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fun = nn.CrossEntropyLoss()
num_epochs = 10

for epoch in range(num_epochs):
    for batch, (X, y) in enumerate(datasets):
        X = X.reshape(-1, 28*28)
        y_hat = model(X)
        loss = loss_fun(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")



