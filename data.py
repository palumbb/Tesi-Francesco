import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_dataset(data_cfg, num_clients, federated: bool, partitioning):
    data_path = data_cfg.path
    if data_path=="./data/consumer.csv":
        dataset, features_ohe, target_name, num_columns = load_consumer()
    elif data_path=="./data/mv.csv":
        dataset, features_ohe, target_name, num_columns = load_mv()
    
    print(dataset)
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)

    train_samples = int(len(dataset)*data_cfg.train_split)
    train = dataset.iloc[0:train_samples,]
    test = dataset.iloc[train_samples:,]
    
    
    scaler = StandardScaler()
    train[num_columns] = scaler.fit_transform(train[num_columns])
    test[num_columns] = scaler.transform(test[num_columns])

    x_train = train[features_ohe].to_numpy()
    x_train = np.vstack(x_train).astype(np.float32)
    y_train = train[target_name].to_numpy()
    y_train = np.vstack(y_train).astype(np.float32)
    x_test = test[features_ohe].to_numpy()
    x_test = np.vstack(x_test).astype(np.float32)
    y_test = test[target_name].to_numpy()
    y_test = np.vstack(y_test).astype(np.float32)
    
    #print(np.unique(y_test, return_counts=True))
    #print(np.unique(y_train, return_counts=True))
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    #print("Train: " + str(train.shape))
    #print("Test: " + str(test.shape))
    #print(train.head)
    #print(test.head)
    #print(dataset.columns)
    #print(dataset.shape)
    
    if federated:
        trainloaders, valloaders, testloader = partition_dataset(num_partitions=num_clients,
                                                                 batch_size=data_cfg.batch_size,
                                                                 val_ratio=data_cfg.val_split,
                                                                 train = train_dataset,
                                                                 test = test_dataset
                                                                 )
        return trainloaders, valloaders, testloader
    
    else:
        return train_dataset, test_dataset

def encoding_categorical_variables(X):
    def encode(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], dummy_na=False, dtype=int)
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return res

    categorical_columns = list(X.select_dtypes(include=['bool','object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode(X,col)
    return X

def partition_dataset(num_partitions: int, batch_size: int, val_ratio: float, train, test):
    #train, test = load_dataset(data_path, train_ratio)
    
    """
    num_instances_per_client = len(train) // num_partitions
    partition_len = [num_instances_per_client] * num_partitions"""

    # change proportions according to num_clients
    # if 2 clients
    #proportions = [.50, .50]
    proportions = [.35, .35, .30]
    lengths = [int(p * len(train)) for p in proportions]
    lengths[-1] = len(train) - sum(lengths[:-1])
    trainsets = random_split(train, lengths)
    #print(len(train))
    #print(partition_len)

    trainloaders = []
    valloaders = []

    #SCALING DEL TRAINING UGUALE ALLO SCALING DEL TESTSET 
    #(CHIEDERE QUALE USARE PER IL TEST SE VENGONO USATI DIVERSI SCALING PER I TRAINING SETS) 

    

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

def load_consumer():
    types = {"ProductID":int, "ProductCategory":str, "ProductBrand":str, "ProductPrice":float,"CustomerAge":float,
            "CustomerGender":str,"PurchaseFrequency":float,"CustomerSatisfaction":float,"PurchaseIntent":int}
    dataset = pd.read_csv("./data/consumer.csv", dtype=types)
    features = list(dataset.columns)
    target_name = "PurchaseIntent"
    features.remove("ProductID")
    target = dataset[target_name]
    #print(np.unique(target, return_counts=True))
    features.remove(target_name)
    num_columns = list(dataset[features].select_dtypes(include=[int, float]).columns)
    dataset = encoding_categorical_variables(dataset[features])
    dataset[target_name] = target
    features_ohe = list(dataset.columns)
    features_ohe.remove(target_name)
    return dataset, features_ohe, target_name, num_columns

def load_mv():
    types = {"x1":float, "x2":float, "x3":str, "x4":float,"x5":float,"x6":float,
                 "x7":str,"x8":str,"x9":float,"x10":float,"binaryClass":str}
    dataset = pd.read_csv("./data/mv.csv", dtype=types)
    features = list(dataset.columns)
    target_name = "binaryClass"
    dataset.replace('N', 0, inplace=True)
    dataset.replace('P', 1, inplace=True)
    target = dataset[target_name]
    features.remove(target_name)
    num_columns = list(dataset[features].select_dtypes(include=[int, float]).columns)
    dataset = encoding_categorical_variables(dataset[features])
    dataset[target_name] = target
    #print(dataset[target_name])
    features_ohe = list(dataset.columns)
    features_ohe.remove(target_name)     
    return dataset, features_ohe, target_name, num_columns
