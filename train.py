import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import model
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchsummary import summary



idx = int(input())
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
plt.imshow(train_set.data[idx], cmap='gray')
plt.show()



def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training ...')
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print('epoch', epoch)
        loss_train = 0.0
        for batch in train_loader:
            imgs, _ = batch  # Unpack the batch into images and labels (ignore labels for autoencoder)
            imgs = imgs.to(device=device)
            imgs = imgs.view(imgs.size(0), -1)  # Flatten each image in the batch
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)

        losses_train.append(loss_train / len(train_loader))

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))
    # model_path = "MLP.8.pth"

    torch.save(model.state_dict(), args.s)
    print(f"Saved trained model to {args.s}")

    # Save the training loss plot
    plt.plot(range(1, n_epochs + 1), losses_train)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.savefig(args.p, format='png')
    plt.show()



if __name__ == "__main__":
    #       read the command in terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', type=int, default=8)
    parser.add_argument('-e', type=int, default=50)
    parser.add_argument('-b', type=int, default=2048)
    parser.add_argument('-s', type=str, default='MLP.8.pth')
    parser.add_argument('-p', type=str, default='loss.MLP.8.png')
    args = parser.parse_args()

    # control frequency between images
    display_interval = 10
    # Initialize the model
    model = model.autoencoderMLP4Layer(N_bottleneck=args.z)

    LEARNINGRATE = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=LEARNINGRATE)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    device = 'cpu'
    loss_fn = nn.MSELoss()

    # Create the data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.b, shuffle=True)
    train(args.e, optimizer, model, loss_fn, train_loader, scheduler, device)

    # call torchSummary here
    summary(model, (1,28*28))







