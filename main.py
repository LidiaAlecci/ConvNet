import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNetsModel(nn.Module):
    def __init__(self, num_classes, cross_entropy_loss=False, kernel_size=3, channel_size1=32, channel_size2=64,
                 dropout=False):
        super(ConvNetsModel, self).__init__()

        self.cross_entropy_loss = cross_entropy_loss
        self.kernel_size = kernel_size
        self.w_dropout = dropout
        self.channel_size1 = channel_size1
        self.channel_size2 = channel_size2
        # input B (batch size),C (channel), H (height), W (width) )
        self.cn1 = nn.Conv2d(in_channels=3, out_channels=channel_size1, kernel_size=(kernel_size, kernel_size))
        self.cn2 = nn.Conv2d(in_channels=channel_size1, out_channels=channel_size1,
                             kernel_size=(kernel_size, kernel_size))
        self.pool = nn.MaxPool2d((2, 2))
        if dropout:
            self.dropout = nn.Dropout(p=0.3)
            self.dropout2 = nn.Dropout(p=0.6)
        self.cn3 = nn.Conv2d(in_channels=channel_size1, out_channels=channel_size2,
                             kernel_size=(kernel_size, kernel_size))
        self.cn4 = nn.Conv2d(in_channels=channel_size2, out_channels=channel_size2,
                             kernel_size=(kernel_size, kernel_size))
        self.new_image_dim = (((32 + 2 * (- kernel_size + 1)) // 2) + 2 * (- kernel_size + 1)) // 2
        self.fc1 = nn.Linear(channel_size2 * self.new_image_dim * self.new_image_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        # No need of the softmax (it is already computed by cross entropy loss)
        if not self.cross_entropy_loss:
            self.softmax = nn.Softmax()

    def forward(self, x):

        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = self.pool(x)
        if self.w_dropout:
            x = self.dropout(x)
        x = F.relu(x)
        x = self.cn3(x)
        x = F.relu(x)
        x = self.cn4(x)
        x = self.pool(x)
        if self.w_dropout:
            x = self.dropout(x)
        x = F.relu(x)
        x = x.view(-1, self.channel_size2 * self.new_image_dim * self.new_image_dim)
        x = self.fc1(x)
        if self.w_dropout:
            x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc2(x)
        # No need of the softmax (it is already computed by cross entropy loss)
        if not self.cross_entropy_loss:
            x = self.softmax(x)
        return x


def show_9_images():
    data = torchvision.datasets.CIFAR10(download=True, root="./data")
    print(f"In the training set there are {len(data)} images")
    fig, ax = plt.subplots(3, 3)
    for i, (image, label) in enumerate(data):
        if i >= 9:
            break
        ax[(i % 3), (i // 3)].imshow(image)
    plt.show()


def CN(num_epochs, batch_size, learning_rate, momentum, kernel_size=3, channel_size1=32, channel_size2=64, result=True,
       dropout=False, plot=True, verbose=True):
    scaled_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True).data / 255

    mean = scaled_data.mean(axis=(0, 1, 2))
    std = scaled_data.std(axis=(0, 1, 2))

    # Transform to normalize each channel
    # In Normalize the first tuple is the mean of the three channels, the second one is the std
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Datasets
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform)

    # Samplers
    split = 49000
    train_sampler = torch.utils.data.SubsetRandomSampler(list(range(split)))
    validation_sampler = torch.utils.data.SubsetRandomSampler(list(range(split, len(train_set))))

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset=train_set, sampler=validation_sampler)

    # Create model
    model = ConvNetsModel(10, True, kernel_size, channel_size1, channel_size2, dropout)
    model = model.to(device)

    # Create loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []
    best_epoch = 0
    best_val_acc = - np.Inf

    # Training
    start_time = time.time()
    for epoch in range(num_epochs):
        batch_loss = []
        batch_val_loss = []
        correct_train = 0
        train_total = 0
        correct_val = 0
        val_total = 0

        model.train()  # Set model in train mode
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()  # reset gradients
            loss.backward()  # compute gradients
            optimizer.step()  # update parameters

            batch_loss.append(loss.item())

            with torch.no_grad():
                _, predicted = outputs.max(1)
            correct_train += (predicted == labels).sum().item()
            train_total += labels.size(0)

            if i % 200 == 0 and verbose:
                # check accuracy.
                print(f'Epoch: {epoch}, steps: {i}, '
                      f'train_loss: {np.average(batch_loss) :.3f}, '
                      f'accuracy: {100 * correct_train / train_total:.1f} %.')

        train_losses.append(np.average(batch_loss))
        train_accuracy.append(100 * correct_train / train_total)

        # Validation for tuning hyper-parameter
        model.eval()
        with torch.no_grad():

            for i_val, (images_val, labels_val) in enumerate(validation_loader):
                images_val = images_val.to(device)  # copy data to GPU
                labels_val = labels_val.to(device)

                outputs_val = model(images_val)
                val_loss = loss_fn(outputs_val, labels_val)
                batch_val_loss.append(val_loss.item())

                _, predicted_val = outputs_val.max(dim=1)

                correct_val += (predicted_val == labels_val).sum().item()
                val_total += labels_val.size(0)

            val_accuracy.append(100 * correct_val / val_total)
            val_losses.append(np.average(batch_val_loss))
            if val_accuracy[-1] > best_val_acc:
                best_val_acc = val_accuracy[-1]
                best_epoch = epoch
    # Evaluation
    model.eval()  # Set model in eval mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
    if result:
        print(f"Time elapsed after {num_epochs} epochs: {round(time.time() - start_time, 4)}")
        print(f"Test accuracy is: {test_acc} %")
        print(f"The best epoch is the {best_epoch}, with a validation accuracy equals to {best_val_acc} %")

    if plot:
        epochs = list(range(num_epochs))
        plot_together(epochs, train_losses, val_losses, train_accuracy, val_accuracy)

    return best_val_acc


def plot_together(epochs, train_losses, val_losses, train_accuracy, val_accuracy):
    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.plot(epochs, val_losses, label="Validation loss")
    ax1.set_xlabel("Epoch", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(epochs, train_accuracy, label="Train accuracy")
    ax2.plot(epochs, val_accuracy, label="Validation accuracy")
    ax2.set_xlabel("Epoch", fontsize=16)
    ax2.set_ylabel("Accuracy", fontsize=16)
    ax2.legend()

    plt.show()


def plot_separate(epochs, train_losses, val_losses, train_accuracy, val_accuracy):
    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, train_losses)

    ax1.set_xlabel("epoch", fontsize=16)
    ax1.set_ylabel("train_loss", fontsize=16)

    fig2, ax2 = plt.subplots()
    ax2.plot(epochs, val_losses)

    ax2.set_xlabel("epoch", fontsize=16)
    ax2.set_ylabel("val_loss", fontsize=16)

    fig3, ax3 = plt.subplots()
    ax3.plot(epochs, train_accuracy)

    ax3.set_xlabel("epoch", fontsize=16)
    ax3.set_ylabel("train_accuracy", fontsize=16)

    fig3, ax3 = plt.subplots()
    ax3.plot(epochs, val_accuracy)

    ax3.set_xlabel("epoch", fontsize=16)
    ax3.set_ylabel("val_accuracy", fontsize=16)

    plt.show()


def run(dropout, verbose=True):
    num_epochs = 20
    batch_size = 32
    learning_rate = 10e-3
    momentum = 0.9
    kernel_size = 3

    CN(num_epochs, batch_size, learning_rate, momentum, kernel_size, dropout=dropout, verbose=verbose)


def find_best_channels_momentum_kernel(dropout=True):
    num_epochs = 20
    batch_size = 32
    learning_rate = 10e-3

    channels_1 = [16, 32, 64, 82]
    channels_2 = [16, 32, 64, 82]
    momentum = [0.7, 0.8, 0.9]
    kernel_sizes = [2, 3, 4, 5]

    best_accuracy = 0
    best_chan1 = 0
    best_chan2 = 0
    best_momentum = 0
    best_kernel = 0
    for chan1 in channels_1:
        for chan2 in channels_2:
            for momentum_ in momentum:
                for kernel_size in kernel_sizes:
                    print(
                        f"Try with chan1 = {chan1}, chan2 = {chan2}, momentum_ = {momentum_}, kernel_size = {kernel_size}")
                    accuracy = CN(num_epochs, batch_size, learning_rate, momentum_, kernel_size, channel_size1=chan1,
                                  channel_size2=chan2, plot=False, dropout=dropout, result=False, verbose=False)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_chan1 = chan1
                        best_chan2 = chan2
                        best_momentum = momentum_
                        best_kernel = kernel_size
    print(f"The best validation accuracy is {best_accuracy} %. "
          f"Best channel 1 value: {best_chan1}."
          f"Best channel 2 value: {best_chan2}."
          f"Best momentum value: {best_momentum}."
          f"Best kernel size: {best_kernel}.")
    CN(num_epochs, batch_size, learning_rate, best_momentum, best_kernel, channel_size1=best_chan1,
       channel_size2=best_chan2, plot=True, dropout=dropout, verbose=True)


def best_run():
    num_epochs = 20
    batch_size = 32
    learning_rate = 10e-3
    best_chan1 = 82
    best_chan2 = 64
    best_momentum = 0.7
    best_kernel = 3
    dropout = True

    CN(num_epochs, batch_size, learning_rate, best_momentum, best_kernel, channel_size1=best_chan1,
       channel_size2=best_chan2, plot=True, dropout=dropout, verbose=True)


if __name__ == '__main__':
    show_9_images()

    run(dropout=False)

    run(dropout=True)

    find_best_channels_momentum_kernel(dropout=True)

    best_run()
