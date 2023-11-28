import torch
import torch.nn 
import torch.nn.functional
import torch.optim
import torchvision

# Note: this requires reshaping EMNIST images from 28x28 to 32x32
class CnnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 47)

    def forward(self, x):
        # out.size() = [batch_size, channels, size, size], -1 here means channels*size*size
        out = torch.nn.functional.relu(self.conv1(x))
        out = torch.nn.functional.avg_pool2d(out, 2)
        out = torch.nn.functional.relu(self.conv2(out))
        out = torch.nn.functional.avg_pool2d(out, 2)

        # Convert to linear layer.
        out = out.view(out.size(0), -1)

        out = torch.nn.functional.relu(self.fc1(out))
        out = torch.nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

minibatch_size = 128
num_epochs = 20
initial_lr = 0.1

def main():
    use_cuda = torch.cuda.is_available()
    print("Using GPU: ", use_cuda)

    # Define the model
    model = CnnModel()
    if use_cuda:
        model = model.cuda()

    # Define the loss function and training algorithm
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

    # Load dataset
    # MNIST_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
    #                                                   torchvision.transforms.ToTensor()])
    # trainset = torchvision.datasets.MNIST(root='./data', train= True, download=True, transform=MNIST_transform)
    # testset = torchvision.datasets.MNIST(root='./data', train= False, download=True, transform=MNIST_transform)

    EMNIST_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
    trainset = torchvision.datasets.EMNIST(root="./data", split="balanced", train=True, transform=EMNIST_transform, download=True)
    testset = torchvision.datasets.EMNIST(root="./data", split="balanced", train=False, transform=EMNIST_transform, download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset)) 

    # Train the model
    for epoch in range(num_epochs):

        optimizer.param_groups[0]["lr"] = compute_learning_rate(epoch)

        # Training Step
        for i, (images, labels) in enumerate(trainloader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            # Forward pass to get the loss
            outputs = model(images) 
            loss = criterion(outputs, labels)

            # Backward and compute the gradient
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
    if epoch < 5:
        return initial_lr
    elif epoch < 10:
        return initial_lr / 10
    else:
        return initial_lr / 100

if __name__ == "__main__":
    main()
