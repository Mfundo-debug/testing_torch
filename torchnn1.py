from turtle import forward, backward
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


train = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
datasets = DataLoader(train,32)
#1, 28, 28 - classes 0-9


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )
    def forward(self, x):
       return self.model(x)

#Instantiate model
clf = ImageClassifier().to("cpu")
#Instantiate loss function
loss_fn = nn.CrossEntropyLoss()
#Instantiate optimizer
optimizer = Adam(clf.parameters(), lr=1e-3)
#Training loop
if __name__ == '__main__':
    with open('model.pth', 'rb') as f:
        clf.load_state_dict(torch.load(f))

    img = Image.open("test.jpeg").convert("L")
    img_tensor = ToTensor()(img).unsqueeze(0).to("cpu")
    pred =torch.argmax(clf(img_tensor))
    print(pred)

    # for epoch in range(10):
    #     for batch in datasets:
    #         x, y = batch
    #         optimizer.zero_grad()
    #         y_pred = clf(x)
    #         loss = loss_fn(y_pred, y)
    #         loss.backward()
    #         optimizer.step()
    #     print(f"Epoch: {epoch} Loss: {loss.item()}")
    # with open("model.pth", "wb") as f:
    #     torch.save(clf.state_dict(), f)