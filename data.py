import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_dataset(data_cfg, num_clients, federated: bool, partitioning):
    data_path = data_cfg.path
    if data_path=="./data/consumer.csv":
        dataset, features_ohe, target_name, num_columns = load_consumer()
    elif data_path=="./data/mv.csv":
        dataset, features_ohe, target_name, num_columns = load_mv()
    
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    #print(dataset)

    if partitioning=="uniform":
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

        # change proportions according to num_clients
        # if 2 clients
        #proportions = [.50, .50]
        proportions = [.35, .35, .30]
        lengths = [int(p * len(train)) for p in proportions]
        lengths[-1] = len(train) - sum(lengths[:-1])
        trainsets = random_split(train, lengths)
    
    elif partitioning=="x3":
        trainsets, test_dataset = split_by_attribute(dataset, num_columns, data_cfg, partitioning, features_ohe, target_name)

    if federated:
            trainloaders, valloaders, testloader = random_partition(num_partitions=num_clients,
                                                                    batch_size=data_cfg.batch_size,
                                                                    val_ratio=data_cfg.val_split,
                                                                    train = trainsets,
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

def random_partition(num_partitions: int, batch_size: int, val_ratio: float, trainsets, test):
    
    """
    num_instances_per_client = len(train) // num_partitions
    partition_len = [num_instances_per_client] * num_partitions"""

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

def split_by_attribute(dataset, num_columns, data_cfg, partitioning, features_ohe, target_name):

    train_samples = int(len(dataset)*data_cfg.train_split)
    train = dataset.iloc[0:train_samples,]
    test = dataset.iloc[train_samples:,]

    scaler = StandardScaler()
    train[num_columns] = scaler.fit_transform(train[num_columns])
    test[num_columns] = scaler.transform(test[num_columns])

    if partitioning == "x3":
        trainsets, brown_ohe, red_ohe,  green_ohe = split_by_x3(train) 

    x_train_subsets = []
    y_train_subsets = []

    for t in trainsets:
        if "x3_brown" in t.columns:
            features_ohe = brown_ohe
        elif "x3_red" in t.columns:
            features_ohe = red_ohe
        elif "x3_green" in t.columns:
            features_ohe = green_ohe

        x_train = t[features_ohe].to_numpy()
        x_train = np.vstack(x_train).astype(np.float32)
        y_train = t[target_name].to_numpy()
        y_train = np.vstack(y_train).astype(np.float32)
        x_train_subsets.append(x_train)
        y_train_subsets.append(y_train)
    

    x_test = test[features_ohe].to_numpy()
    x_test = np.vstack(x_test).astype(np.float32)
    y_test = test[target_name].to_numpy()
    y_test = np.vstack(y_test).astype(np.float32)
        
    x_train_tensor_subsets = []
    y_train_tensor_subsets = []
    
    for x in x_train_subsets:
        x_train_tensor = torch.tensor(x, dtype=torch.float32)
        x_train_tensor_subsets.append(x_train_tensor)
    print(x_train_tensor_subsets)
    #x_train_tensor_subsets = np.asarray(x_train_tensor_subsets)
        
    for y in y_train_subsets:
        y_train_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        y_train_tensor_subsets.append(y_train_tensor)
    #y_train_tensor_subsets = np.asarray(y_train_tensor_subsets)

    #x_train_tensor_subsets = np.array(x_train_tensor_subsets)
    #y_train_tensor_subsets = np.array(y_train_tensor_subsets)
    x_train_tensor_subsets = torch.tensor(x_train_tensor_subsets, dtype=torch.float32).view(-1, 1)
    y_train_tensor_subsets = torch.tensor(y_train_tensor_subsets, dtype=torch.float32).view(-1, 1)

    x_test_tensor = torch.tensor(x, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


    train_dataset = TensorDataset(x_train_tensor_subsets,y_train_tensor_subsets)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    return train_dataset, test_dataset


def split_by_x3(df):

    subset_brown = df[df['x3_brown'] == 1][['x1', 'x2', 'x4', 'x5', 'x6', 'x9', 'x10', 'x3_brown', 'x7_no', 'x7_yes', 'x8_large', 'x8_normal', 'binaryClass']]
    brown_ohe = ['x1', 'x2', 'x4', 'x5', 'x6', 'x9', 'x10', 'x3_brown', 'x7_no', 'x7_yes', 'x8_large', 'x8_normal', 'binaryClass']
    subset_red = df[df['x3_red'] == 1][['x1', 'x2', 'x4', 'x5', 'x6', 'x9', 'x10', 'x3_red', 'x7_no', 'x7_yes', 'x8_large', 'x8_normal', 'binaryClass']]
    red_ohe = ['x1', 'x2', 'x4', 'x5', 'x6', 'x9', 'x10', 'x3_red', 'x7_no', 'x7_yes', 'x8_large', 'x8_normal', 'binaryClass']
    subset_green = df[df['x3_green'] == 1][['x1', 'x2', 'x4', 'x5', 'x6', 'x9', 'x10', 'x3_green', 'x7_no', 'x7_yes', 'x8_large', 'x8_normal', 'binaryClass']]
    green_ohe = ['x1', 'x2', 'x4', 'x5', 'x6', 'x9', 'x10', 'x3_green', 'x7_no', 'x7_yes', 'x8_large', 'x8_normal', 'binaryClass']

    subsets = [subset_brown, subset_red, subset_green]

    return subsets, brown_ohe, red_ohe, green_ohe
