import matplotlib.pyplot as plt
import torch
from tqdm import trange
from matplotlib import pyplot as plt
import os
import json


def num_of_params(model, only_trainable: bool = False): # good function from StackOverflow
    parameters = list(model.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def get_acc_and_loss(model, device, dataloader, loss_function):
    with torch.no_grad():
        correct_predictions = 0
        accumulated_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.type(torch.LongTensor).to(device)
            model_outputs = model(inputs)
            accumulated_loss += float(loss_function(model_outputs, targets))
            predicted_classes = model_outputs.argmax(dim=-1)
            correct_predictions += int((predicted_classes == targets).sum())
    return correct_predictions / len(dataloader.dataset), accumulated_loss / len(dataloader.dataset)

def make_checkpoint(model, optimizer, statistics_dict, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'statistics': statistics_dict}, dir_path+"/ep"+str(len(statistics_dict["trainacc"]) - 1)+".pt")

def train(model, device, optimizer, loss_function, trainloader, testloader, statistics_dict, epochs_n=10,
          augmenter=None, checkpoints_dir=None, checkpoints_per=10):
    
    if ((len(statistics_dict["testloss"]) - 1) % checkpoints_per == 0) and (checkpoints_dir != None):
            make_checkpoint(model, optimizer, statistics_dict, dir_path=checkpoints_dir)
    
    for epoch in trange(epochs_n):
        model.train()
        correct_predictions = 0 # it's need to calculate acc.
        accumulated_loss = 0
        for inputs, targets in trainloader:
            if augmenter != None:
                inputs = augmenter(inputs)
            inputs = inputs.to(device)
            targets = targets.type(torch.LongTensor).to(device)
            
            model_outputs = model(inputs)
            loss = loss_function(model_outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            accumulated_loss += float(loss)
            predicted_classes = model_outputs.argmax(dim=-1)
            correct_predictions += int((predicted_classes == targets).sum())
        model.eval()
        
        test_acc, test_loss = get_acc_and_loss(model, device, testloader, loss_function)
        statistics_dict["trainacc"].append(correct_predictions / len(trainloader.dataset))
        statistics_dict["trainloss"].append(accumulated_loss / len(trainloader.dataset))
        statistics_dict["testacc"].append(test_acc)
        statistics_dict["testloss"].append(test_loss)
    
        if ((len(statistics_dict["testloss"]) - 1) % checkpoints_per == 0) and (checkpoints_dir != None):
            make_checkpoint(model, optimizer, statistics_dict, dir_path=checkpoints_dir)
            
def init_statistics(model, device, trainloader, testloader, loss_function):
    train_init_acc, train_init_loss =  get_acc_and_loss(model, device, trainloader, loss_function)
    test_init_acc, test_init_loss =  get_acc_and_loss(model, device, testloader, loss_function)
    statistics_dict = {"trainacc": [train_init_acc],
                  "testacc": [test_init_acc],
                  "trainloss": [train_init_loss],
                  "testloss": [test_init_loss],
                 }
    return statistics_dict


def add_statistics_to_json(model_info, statistics_dict, path_dir, filename):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    
    path = path_dir + "/" + filename
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump([{"model_info": model_info, "statistics": statistics_dict}], f)
    else:
        with open(path) as f:
            old_data = json.load(f)
        with open(path, 'w') as f:
            json.dump(old_data+[{"model_info": model_info, "statistics": statistics_dict}], f)

        
def plot_statistics(statistics_dict):
    plt.plot(statistics_dict["trainacc"], color='green', label="train accuracy")
    plt.plot(statistics_dict["testacc"], color='red', label="test accuracy")
    plt.legend()
    plt.show()
    
    plt.plot(statistics_dict["trainloss"], color='green', label="train loss")
    plt.plot(statistics_dict["testloss"], color='red', label="test loss")
    plt.legend()
    plt.show()
