import torch
import torch.nn 
import torch.nn.functional
import torch.optim
import torch.utils.data
import torchvision

class CnnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 50, 5)
        self.conv2 = torch.nn.Conv2d(50, 240, 5)
        self.fc1 = torch.nn.Linear(3840, 6000)
        self.fc2 = torch.nn.Linear(6000, 1000)
        self.fc3 = torch.nn.Linear(1000, 47)

    def forward(self, x):
        out = torch.nn.functional.relu(self.conv1(x))
        out = torch.nn.functional.max_pool2d(out, 2)
        out = torch.nn.functional.relu(self.conv2(out))
        out = torch.nn.functional.max_pool2d(out, 2)

        # Convert to linear layer
        # out.size() = [batch_size, channels, size, size], -1 here means channels*size*size
        out = out.view(out.size(0), -1)

        out = torch.nn.functional.relu(self.fc1(out))
        out = torch.nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

minibatch_size = 32
num_epochs = 20
initial_lr = 0.1

def main():
    use_cuda = torch.cuda.is_available()
    print(f"Using GPU: {use_cuda}")

    # Define the model
    model = CnnModel()
    if use_cuda:
        model = model.cuda()
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define the loss function and training algorithm
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

    # Load dataset
    EMNIST_transform = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.EMNIST(root="./data", split="balanced", train=True, transform=EMNIST_transform, download=True)
    testset = torchvision.datasets.EMNIST(root="./data", split="balanced", train=False, transform=EMNIST_transform, download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    # Train the model
    for epoch in range(num_epochs):

        for group in optimizer.param_groups:
            group["lr"] = compute_learning_rate(epoch)

        # Training Step
        for i, (images, labels) in enumerate(trainloader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            # Forward pass to get the loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()  # backpropragation
            optimizer.step() # update the weights/parameters

        # Training accuracy
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(trainloader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            p_max, predicted = torch.max(outputs, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum()
        training_accuracy = float(correct) / total

        # Test accuracy
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(testloader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            p_max, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        test_accuracy = float(correct) / total

        print(f"Epoch: {epoch+1}, training accuracy: {training_accuracy:2f}, test accuracy: {test_accuracy:2f}")


def compute_learning_rate(epoch):
    if epoch < 10:
        return initial_lr
    elif epoch < 15:
        return initial_lr / 10
    else:
        return initial_lr / 100


if __name__ == "__main__":
    main()
