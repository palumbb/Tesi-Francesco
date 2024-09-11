import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor


#data_path = "./data/mv.csv"
#train_ratio = 0.8

def load_dataset(data_path, train_ratio):
    dataset = pd.read_csv(data_path)
    #tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_samples = int(len(dataset)*train_ratio)
    train = dataset.iloc[0:train_samples,]
    test = dataset.iloc[train_samples:,]
    #print("Train: " + str(train.shape))
    #print("Test: " + str(test.shape))
    #print(train.head)
    #print(test.head)
    return train, test
    

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float, train, test):
    #train, test = load_dataset(data_path, train_ratio)
    
    num_instances_per_client = len(train) // num_partitions
    partition_len = [num_instances_per_client] * num_partitions

    #print(len(train))
    #print(partition_len)

    trainsets = random_split(
        train, partition_len, torch.Generator().manual_seed(2023)
    )

    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )
    
    testloader = DataLoader(test, batch_size=128)

    #print("Partition done successfully")

    return trainloaders, valloaders, testloader