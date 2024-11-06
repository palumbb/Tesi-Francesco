import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn import preprocessing
import numpy as np
from data_handler.quality import impute_missing_column, dirty

def load_dataset(data_cfg, num_clients, federated: bool, partitioning, quality, model, dirty_percentage):
    data_path = data_cfg.path
    if quality=="completeness":
        trainsets, test_dataset = load_dirty_dataset(data_path, num_clients, dirty_percentage, data_cfg)
        trainloaders, valloaders, testloader = data_loaders(num_partitions=num_clients,
                                                                    batch_size=data_cfg.batch_size,
                                                                    val_ratio=data_cfg.val_split,
                                                                    train = trainsets,
                                                                    test = test_dataset
                                                                    )
        return trainloaders, valloaders, testloader
    elif quality=="normal":
        dataset, features_ohe, target_name, num_columns, num_classes, to_view = load_clean_dataset(data_path, model)
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)

        train_dataset, test_dataset = train_test_split(dataset, data_cfg, num_columns, features_ohe, target_name, num_classes, to_view)
        
        if federated:
            if partitioning=="uniform":
                if num_clients == 2:
                    proportions = [.50, .50]
                elif num_clients == 3:
                    proportions = [.35, .35, .30]
                elif num_clients == 10:
                    proportions = np.ones(10)*0.10
                elif num_clients == 50:
                    proportions = np.ones(50)*0.02
                elif num_clients == 100:
                    proportions = np.ones(100)*0.01
                elif num_clients == 1000:
                    proportions = np.ones(1000)*0.001
                
                lengths = [int(p * len(train_dataset)) for p in proportions]
                lengths[-1] = len(train_dataset) - sum(lengths[:-1])
                trainsets = random_split(train_dataset, lengths)
            else:
                trainsets, test_dataset = split_by_attribute(dataset, num_columns, data_cfg, partitioning, features_ohe, target_name, to_view, num_clients, dirty_percentage)

            trainloaders, valloaders, testloader = data_loaders(num_partitions=num_clients,
                                                                    batch_size=data_cfg.batch_size,
                                                                    val_ratio=data_cfg.val_split,
                                                                    train = trainsets,
                                                                    test = test_dataset
                                                                    )
            return trainloaders, valloaders, testloader
        
        else:
            return train_dataset, test_dataset
    
    
def load_clean_dataset(data_path, model):
    if data_path=="./datasets/consumer.csv":
        if (model == "model.multiclassnet.MulticlassNet"):
            dataset, features_ohe, target_name, num_columns = load_consumer_binary()
            to_view = True
        else:
            dataset, features_ohe, target_name, num_columns = load_consumer_multi()
            to_view = False
        num_classes = 2
    elif data_path=="./datasets/mv.csv":
        if (model == "model.multiclassnet.MulticlassNet"):
            dataset, features_ohe, target_name, num_columns = load_mv_binary()
            to_view = True
        else:
            dataset, features_ohe, target_name, num_columns = load_mv_multi()
            to_view = False
        num_classes = 2
    elif data_path=="./datasets/car.csv":
        dataset, features_ohe, target_name, num_columns = load_car()
        num_classes = 4
        to_view = False
    elif data_path=="./datasets/nursery.csv":
        dataset, features_ohe, target_name, num_columns = load_nursery()
        num_classes = 5
        to_view = False
    elif data_path=="./datasets/shuttle.csv":
        dataset, features_ohe, target_name, num_columns = load_shuttle()
        num_classes = 7
        to_view = False
    elif data_path=="./datasets/wall-robot-navigation.csv":
        dataset, features_ohe, target_name, num_columns = load_wall()
        num_classes = 4
        to_view = False
    elif data_path=="./datasets/mushrooms.csv":
        dataset, features_ohe, target_name, num_columns = load_mushrooms()
        num_classes = 2
        to_view = False
    return dataset, features_ohe, target_name, num_columns, num_classes, to_view

def get_data_info(data_path):
    if data_path=="./datasets/consumer.csv":
        types = {"ProductID":int, "ProductCategory":str, "ProductBrand":str, "ProductPrice":float,"CustomerAge":float,
            "CustomerGender":str,"PurchaseFrequency":float,"CustomerSatisfaction":float,"PurchaseIntent":int}
        target_encoded = ["PurchaseIntent_N", "PurchaseIntent_Y"]
        target= "PurchaseIntent"
    elif data_path=="./datasets/mv.csv":
        types = {"x1":float, "x2":float, "x3":str, "x4":float,"x5":float,"x6":float,
                 "x7":str,"x8":str,"x9":float,"x10":float,"binaryClass":str}
        target_encoded = ["binaryClass_N", "binaryClass_P"]
        target = "binaryClass"
    elif data_path=="./datasets/car.csv":
        types = {"index":str, "buying":str, "maint":str, "doors":str,"persons":str,
            "lug_boot":str,"safety":str}
        target_encoded = ['safety_acc',  'safety_good',  'safety_unacc',  'safety_vgood']
        target = "safety"
    elif data_path=="./datasets/nursery.csv":
        types = {"parents":str, "has_nurs":str, "form":str, "children":int,"housing":str,
            "finance":str,"social":str, "health":str, "class":str}
        target_encoded = ["'class'_not_recom", "'class'_priority", "'class'_recommend", "'class'_spec_prior", "'class'_very_recom"]
        target = "'class'"
    elif data_path=="./datasets/mushrooms.csv":
        types = {'CapShape' : str, 'CapSurface' : str, 'CapColor' : str, 'Bruises' : bool, 'Odor' : str,
       'GillAttachment' : str, 'GillSpacing' : str, 'GillSize' : str, 'GillColor' : str, 'StalkShape' : str,
       'StalkRoot' : str, 'StalkSurfaceAboveRing' : str, 'StalkSurfaceBelowRing' : str,
       'StalkColorAboveRing' : str, 'StalkColorBelowRing' : str, 'VeilType' : str, 'VeilColor' : str,
       'RingNumber' : str, 'RingType' : str, 'SporePrintColor' : str, 'Population' : str, 'Habitat' : str,
       'Class' : str}
        target_encoded = ["Class_poisonous", "Class_edible"]
        target = "Class"
    elif data_path=="./datasets/shuttle.csv":
        types = {'A1': int, 'A2': int, 'A3': int, 'A4': int, 'A5': int, 'A6': int, 'A7': int, 'A8': int, 'A9': int, 'class': str}
        target_encoded = ["class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7"]
        target = "class"
    elif data_path=="./datasets/wall-robot-navigation.csv":
        types = {'V1': float, 'V2': float, 'V3': float, 'V4': float, 'Class': str}
        target_encoded = ["Class_1","Class_2","Class_3","Class_4"]
        target = "Class"
    return types, target_encoded, target
        
def load_dirty_dataset(data_path, num_clients, dirty_percentage, data_cfg):
    types, target_encoded, target = get_data_info(data_path)
    df = pd.read_csv(data_path, dtype=types)
    if data_path=="./datasets/consumer.csv":
        df['PurchaseIntent'].replace(0, 'N', inplace=True)
        df['PurchaseIntent'].replace(1, 'Y', inplace=True)
    train_samples = int(len(df)*data_cfg.train_split)
    train = df.iloc[0:train_samples,]
    test = df.iloc[train_samples:,]
    features = list(df.columns)
    features.remove(target)
    to_view = False # True if BinaryNet, False otherwise
    method = "uniform"
    seed = 0
    if num_clients == 2:
        proportions = 0.5
    elif num_clients == 5:
        proportions = 0.2
    elif num_clients == 10:
        proportions = 0.1
    elif num_clients == 50:
        proportions = 0.02
    elif num_clients == 100:
        proportions = 0.01
    percentages = np.ones(num_clients)*proportions
    subsets = split_dataframe(train, percentages, num_clients)
    dirty_clients = []
    for s in subsets:
        client = dirty(seed, s, features, method, dirty_percentage)
        dirty_clients.append(client)
    dirty_clients, test = one_hot_encode_dirty(dirty_clients, test)
    print(dirty_clients[0].head(20))
    features_ohe = list(test.columns)
    for t in target_encoded:
        features_ohe.remove(t)
    train_datasets, test_dataset = get_train_test(dirty_clients, test, features_ohe, target_encoded, to_view)
    return train_datasets, test_dataset

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

def one_hot_encode_dirty(subsets, test):
    categorical_columns = subsets[0].select_dtypes(include=['object', 'category']).columns
    encoded_subsets = []
    for subset in subsets:
        encoded_subset = pd.get_dummies(subset, columns=categorical_columns)
        encoded_subsets.append(encoded_subset)

    encoded_test_set = pd.get_dummies(test, columns=categorical_columns)

    all_columns = set().union(*[df.columns for df in encoded_subsets], encoded_test_set.columns)

    aligned_subsets = [df.reindex(columns=all_columns, fill_value=0) for df in encoded_subsets]
    aligned_test_set = encoded_test_set.reindex(columns=all_columns, fill_value=0)
    
    return aligned_subsets, aligned_test_set

def data_loaders(num_partitions: int, batch_size: int, val_ratio: float, train, test):
    
    """
    num_instances_per_client = len(train) // num_partitions
    partition_len = [num_instances_per_client] * num_partitions
    """

    trainloaders = []
    valloaders = []

    #SCALING DEL TRAINING UGUALE ALLO SCALING DEL TESTSET 
    #(CHIEDERE QUALE USARE PER IL TEST SE VENGONO USATI DIVERSI SCALING PER I TRAINING SETS) 

    for trainset_ in train:
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

    testloader = DataLoader(test, batch_size=batch_size)

    return trainloaders, valloaders, testloader

def load_consumer_binary():
    # MIXED
    types = {"ProductID":int, "ProductCategory":str, "ProductBrand":str, "ProductPrice":float,"CustomerAge":float,
            "CustomerGender":str,"PurchaseFrequency":float,"CustomerSatisfaction":float,"PurchaseIntent":int}
    dataset = pd.read_csv("./datasets/consumer.csv", dtype=types)
    features = list(dataset.columns)
    target_name = "PurchaseIntent"
    features.remove("ProductID")
    target = dataset[target_name]
    features.remove(target_name)
    num_columns = list(dataset[features].select_dtypes(include=[int, float]).columns)
    profiling(dataset, "./datasets/consumer.csv")
    dataset = encoding_categorical_variables(dataset[features])
    dataset[target_name] = target
    features_ohe = list(dataset.columns)
    features_ohe.remove(target_name)
    return dataset, features_ohe, target_name, num_columns

def load_consumer_multi():
    types = {"ProductID":int, "ProductCategory":str, "ProductBrand":str, "ProductPrice":float,"CustomerAge":float,
            "CustomerGender":str,"PurchaseFrequency":float,"CustomerSatisfaction":float,"PurchaseIntent":int}
    dataset = pd.read_csv("./datasets/consumer.csv", dtype=types)
    features = list(dataset.columns)
    target_name = ["PurchaseIntent_N", "PurchaseIntent_Y"]
    features.remove("ProductID")
    dataset['PurchaseIntent'].replace(0, 'N', inplace=True)
    dataset['PurchaseIntent'].replace(1, 'Y', inplace=True)
    num_columns = list(dataset[features].select_dtypes(include=[int, float]).columns)
    #profiling(dataset, "./datasets/consumer.csv")
    dataset = encoding_categorical_variables(dataset[features])
    features_ohe = list(dataset.columns)
    for t in target_name:
        features_ohe.remove(t)
    return dataset, features_ohe, target_name, num_columns

def load_mv_binary():
    # MIXED
    types = {"x1":float, "x2":float, "x3":str, "x4":float,"x5":float,"x6":float,
                 "x7":str,"x8":str,"x9":float,"x10":float,"binaryClass":str}
    dataset = pd.read_csv("./datasets/mv.csv", dtype=types)
    features = list(dataset.columns)
    target_name = "binaryClass"
    dataset.replace('N', 0, inplace=True)
    dataset.replace('P', 1, inplace=True)
    target = dataset[target_name]
    features.remove(target_name)
    num_columns = list(dataset[features].select_dtypes(include=[int, float]).columns)
    profiling(dataset, "./datasets/mv.csv")
    dataset = encoding_categorical_variables(dataset[features])
    dataset[target_name] = target
    features_ohe = list(dataset.columns)
    features_ohe.remove(target_name)
    return dataset, features_ohe, target_name, num_columns

def load_mv_multi():
    types = {"x1":float, "x2":float, "x3":str, "x4":float,"x5":float,"x6":float,
                 "x7":str,"x8":str,"x9":float,"x10":float,"binaryClass":str}
    dataset = pd.read_csv("./datasets/mv.csv", dtype=types)
    features = list(dataset.columns)
    target_name = ["binaryClass_N", "binaryClass_P"]
    num_columns = list(dataset[features].select_dtypes(include=[int, float]).columns)
    #profiling(dataset, "./datasets/mv.csv")
    dataset = encoding_categorical_variables(dataset[features])
    features_ohe = list(dataset.columns)
    for t in target_name:
        features_ohe.remove(t)
    return dataset, features_ohe, target_name, num_columns

def load_car():
    # CATEGORICAL
    types = {"index":str, "buying":str, "maint":str, "doors":str,"persons":str,
            "lug_boot":str,"safety":str}
    dataset = pd.read_csv("./datasets/car.csv", dtype=types)
    features = list(dataset.columns)
    target_name = ['safety_acc',  'safety_good',  'safety_unacc',  'safety_vgood']
    profiling(dataset, "./datasets/car.csv")
    dataset = encoding_categorical_variables(dataset[features])
    features_ohe = list(dataset.columns)
    num_columns = 'categorical'
    for t in target_name:
        features_ohe.remove(t)
    return dataset, features_ohe, target_name, num_columns

def load_nursery():
    # MIXED
    types = {"parents":str, "has_nurs":str, "form":str, "children":int,"housing":str,
            "finance":str,"social":str, "health":str, "class":str}
    dataset = pd.read_csv("./datasets/nursery.csv", dtype=types)
    features = list(dataset.columns)
    target_name = ["'class'_not_recom", "'class'_priority", "'class'_recommend", "'class'_spec_prior", "'class'_very_recom"]
    profiling(dataset, "./datasets/nursery.csv")    
    dataset = encoding_categorical_variables(dataset[features])
    features_ohe = list(dataset.columns)
    num_columns = 'categorical'
    for t in target_name:
        features_ohe.remove(t)
    return dataset, features_ohe, target_name, num_columns

def load_mushrooms():
    # CATEGORICAL
    types = {'CapShape' : str, 'CapSurface' : str, 'CapColor' : str, 'Bruises' : bool, 'Odor' : str,
       'GillAttachment' : str, 'GillSpacing' : str, 'GillSize' : str, 'GillColor' : str, 'StalkShape' : str,
       'StalkRoot' : str, 'StalkSurfaceAboveRing' : str, 'StalkSurfaceBelowRing' : str,
       'StalkColorAboveRing' : str, 'StalkColorBelowRing' : str, 'VeilType' : str, 'VeilColor' : str,
       'RingNumber' : str, 'RingType' : str, 'SporePrintColor' : str, 'Population' : str, 'Habitat' : str,
       'Class' : str}
    dataset = pd.read_csv("./datasets/mushrooms.csv", dtype=types)
    features = list(dataset.columns)
    target_name = ["Class_poisonous", "Class_edible"]
    profiling(dataset, "./datasets/mushrooms.csv")
    dataset = encoding_categorical_variables(dataset[features])
    features_ohe = list(dataset.columns)
    num_columns = 'categorical'
    for t in target_name:
        features_ohe.remove(t)
    return dataset, features_ohe, target_name, num_columns

def load_shuttle():
    # NUMERICAL
    types = {'A1': int, 'A2': int, 'A3': int, 'A4': int, 'A5': int, 'A6': int, 'A7': int, 'A8': int, 'A9': int, 'class': str}
    dataset = pd.read_csv("./datasets/shuttle.csv", dtype=types)
    features = list(dataset.columns)
    target_name = ["class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7"]
    num_columns = list(dataset[features].select_dtypes(include=[int, float]).columns)
    profiling(dataset, "./datasets/shuttle.csv")
    dataset = encoding_categorical_variables(dataset[features])
    features_ohe = list(dataset.columns)
    for t in target_name:
        features_ohe.remove(t)
    return dataset, features_ohe, target_name, num_columns

def load_wall():
    # NUMERICAL
    types = {'V1': float, 'V2': float, 'V3': float, 'V4': float, 'Class': str}
    dataset = pd.read_csv("./datasets/wall-robot-navigation.csv", dtype=types)
    features = list(dataset.columns)
    target_name = ["Class_1","Class_2","Class_3","Class_4"]
    num_columns = list(dataset[features].select_dtypes(include=[int, float]).columns)
    profiling(dataset, "./datasets/wall-robot-navigation.csv")
    dataset = encoding_categorical_variables(dataset[features])
    features_ohe = list(dataset.columns)
    for t in target_name:
        features_ohe.remove(t)
    return dataset, features_ohe, target_name, num_columns

def profiling(df, data_path):
    data = df.copy()
    le = preprocessing.LabelEncoder()
    data = data.apply(le.fit_transform) 
    select_features(data, data_path)
    """print(f"Null Values:\n {df.isna().sum()}")
    print("\nUnique Values:")
    for col in df:
        print(f"{col} : {df[col].unique()}")
    plt.figure(figsize=(10,12))
    #print(data.head(10))
    cor = data.corr(method='kendall')
    sns.heatmap(cor, xticklabels=True, yticklabels=True, annot=True, cmap=plt.cm.Reds)
    plt.show()
    #compute_associationrules(df, data)
    #select_features"""

def select_features(df, data_path):
    X = df.copy()
    if data_path == "./datasets/car.csv":
        X.drop('safety', axis=1)
        y = X["safety"]
    elif data_path == "./datasets/shuttle.csv":
        X.drop('class', axis=1)
        y = X["class"]
    elif data_path == "./datasets/consumer.csv":
        X.drop('PurchaseIntent', axis=1)
        y = X["PurchaseIntent"]
    elif data_path == "./datasets/nursery.csv":
        X.drop("'class'", axis=1)
        y = X["'class'"]
    elif data_path == "./datasets/mv.csv":
        X.drop('binaryclass', axis=1)
        y = X["binaryclass"]
    elif data_path == "./datasets/wall-robot-navigation.csv":
        X.drop('Class', axis=1)
        y = X["Class"]
    elif data_path == "./datasets/mushrooms.csv":
        X.drop('Class', axis=1)
        y = X["Class"]
    print(X.shape)
    selector = SelectKBest(mutual_info_classif, k='all')
    selector.fit(X, y)
    scores = selector.scores_
    indexes = np.argsort(scores)[::-1][:3]

    print(X.columns[indexes])
    print(scores[indexes])
    
def compute_associationrules(df, data):
    if data == "./datasets/mv.csv":
        association_cols = ['x3_brown',  'x3_green',  'x3_red',  'x7_no',  'x7_yes',  'x8_large',  'x8_normal', 'binaryClass']
        rules = apriori(df[association_cols], min_support = 0.2, use_colnames = True, verbose = 1)
        rules = rules.set_index('itemsets').filter(like='binaryClass', axis=0)
    elif data == "./datasets/consumer.csv":
        association_cols = ['ProductCategory_Headphones',
       'ProductCategory_Laptops', 'ProductCategory_Smart Watches',
       'ProductCategory_Smartphones', 'ProductCategory_Tablets',
       'ProductBrand_Apple', 'ProductBrand_HP', 'ProductBrand_Other Brands',
       'ProductBrand_Samsung', 'ProductBrand_Sony', 'CustomerGender_0',
       'CustomerGender_1', 'PurchaseIntent']
        rules = apriori(df[association_cols], min_support = 0.2, use_colnames = True, verbose = 1)
        rules = rules.set_index('itemsets').filter(like='PurchaseIntent', axis=0)
    elif data == "./datasets/car.csv":
        association_cols = ['index_high', 'index_low', 'index_med', 'index_vhigh', 'buying_high',
       'buying_low', 'buying_med', 'buying_vhigh', 'maint_2', 'maint_3',
       'maint_4', 'maint_5more', 'doors_2', 'doors_4', 'doors_more',
       'persons_big', 'persons_med', 'persons_small', 'lug_boot_high',
       'lug_boot_low', 'lug_boot_med', 'safety_unacc', 'safety_good', 'safety_acc', 'safety_vgood']
        rules = apriori(df[association_cols], min_support = 0.29, use_colnames = True, verbose = 1)
        rules = rules.set_index('itemsets').filter(like='safety', axis=0)
    elif data == "./datasets/nursery.csv":
        return 0
    print(rules)
    
def train_test_split(dataset, data_cfg, num_columns, features_ohe, target_name, num_classes, to_view):
    # ONLY USED FOR RANDOM SPLIT
    train_samples = int(len(dataset)*data_cfg.train_split)
    train = dataset.iloc[0:train_samples,]
    test = dataset.iloc[train_samples:,]
    if (num_columns != 'categorical'):
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

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    if to_view == True:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    else:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    return train_dataset, test_dataset

def split_by_attribute(dataset, num_columns, data_cfg, partitioning, features_ohe, target_name, to_view, num_clients, dirty_percentage):
    # ONLY USED FOR ATTRIBUTE SPLIT
    train_samples = int(len(dataset)*data_cfg.train_split)
    train = dataset.iloc[0:train_samples,]
    test = dataset.iloc[train_samples:,]
    if num_columns != 'categorical':
        scaler = StandardScaler()
        train[num_columns] = scaler.fit_transform(train[num_columns])
        test[num_columns] = scaler.transform(test[num_columns])

    if partitioning == "x3":
        train_list = split_by_x3(train)
    elif partitioning == "x4":
        train_list = split_by_x4(train)
    elif partitioning == "x5":
        train_list = split_by_x5(train)
    elif partitioning == "x6":
        train_list = split_by_x6(train)
    elif partitioning == "x8":
        train_list = split_by_x8(train)
    elif partitioning == "brand":
        train_list = split_by_brand(train)
    elif partitioning == "category":
        train_list = split_by_category(train)
    elif partitioning == "x10":
        train_list = split_by_x10(train)
    elif partitioning == "age":
        train_list = split_by_age(train)
    elif partitioning == "gender":
        train_list = split_by_gender(train)
    elif partitioning == "satisfaction":
        train_list = split_by_satisfaction(train)
    elif partitioning == "doors":
        train_list = split_by_doors(train)
    elif partitioning == "health":
        train_list = split_by_health(train)
    elif partitioning == "odor":
        train_list = split_by_odor(train)
    elif partitioning == "a1":
        train_list = split_by_a1(train)
    elif partitioning == "v1":
        train_list = split_by_v1(train)
    elif partitioning == "v2":
        train_list = split_by_v2(train)
    
    train_datasets, test_dataset = get_train_test(train_list, test, features_ohe, target_name, to_view)
    return train_datasets, test_dataset

def get_train_test(train_list, test, features_ohe, target_name, to_view):
    x_train_list = []
    y_train_list = []
    for train_df in train_list:
        x_train = train_df[features_ohe].to_numpy()
        x_train = np.vstack(x_train).astype(np.float32)
        y_train = train_df[target_name].to_numpy()
        y_train = np.vstack(y_train).astype(np.float32)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        if to_view ==True:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        x_train_list.append(x_train_tensor)
        y_train_list.append(y_train_tensor)
    
    x_test = test[features_ohe].to_numpy()
    x_test = np.vstack(x_test).astype(np.float32)
    y_test = test[target_name].to_numpy()
    y_test = np.vstack(y_test).astype(np.float32)
    
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    if to_view==True:
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    else:
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_datasets = [TensorDataset(x_tensor, y_tensor) for x_tensor, y_tensor in zip(x_train_list, y_train_list)]
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    return train_datasets, test_dataset

def split_dataframe(df, percentages, num_clients):
    if len(percentages) != num_clients:
        raise ValueError("Il numero di percentuali deve essere uguale a num_clients.")
    if sum(percentages) != 1:
        raise ValueError("La somma delle percentuali deve essere pari a 1.")
    subsets = []
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + int(percentages[i] * len(df))
        subsets.append(df.iloc[start_idx:end_idx])
        start_idx = end_idx
    return subsets


def split_by_x3(df):
    cols = list(df.columns)
    subset_brown = df[df['x3_brown'] == 1][cols]
    subset_red = df[df['x3_red'] == 1][cols]
    subset_green = df[df['x3_green'] == 1][cols]
    
    subsets = [subset_brown, subset_red, subset_green]
    return subsets

def split_by_x4(df):
    cols = list(df.columns)
    subset_1 = df[(df['x4'] <= 0.0) & (df['x4'] > -0.5)][cols]
    subset_2= df[(df['x4'] > 0.0) & (df['x4'] <= 1)][cols]
    subset_3 = df[(df['x4'] > 1)][cols]
    subset_4 = df[df['x4'] <= -0.5][cols]
    
    subsets = [subset_1, subset_2, subset_3, subset_4]
    
    return subsets

def split_by_x5(df):
    cols = list(df.columns)
    subset_1 = df[(df['x5'] <= 0.0) & (df['x5'] > -0.5)][cols]
    subset_2= df[(df['x5'] > 0.0) & (df['x5'] <= 1)][cols]
    subset_3 = df[(df['x5'] > 1)][cols]
    subset_4 = df[df['x5'] <= -0.5][cols]

    subsets = [subset_1, subset_2, subset_3, subset_4]
    return subsets

def split_by_x6(df):
    cols = list(df.columns)
    subset_1 = df[(df['x6'] <= 0.0) & (df['x6'] > -0.5)][cols]
    subset_2= df[(df['x6'] > 0.0) & (df['x6'] <= 1)][cols]
    subset_3 = df[(df['x6'] > 1)][cols]
    subset_4 = df[df['x6'] <= -0.5][cols]

    subsets = [subset_1, subset_2, subset_3, subset_4]
    return subsets

def split_by_x8(df):
    cols = list(df.columns)
    subset_normal = df[df['x8_normal'] == 1][cols]
    subset_large = df[df['x8_large'] == 1][cols]

    subsets = [subset_normal, subset_large]
    return subsets

def split_by_x10(df):
    cols = list(df.columns)
    subset_neg = df[df['x10'] <= 0.0][cols]
    subset_range1= df[(df['x10'] > 0.0) & (df['x10'] <= 0.5)][cols]
    subset_range2 = df[(df['x10'] > 0.5) & (df['x10'] <= 1.0)][cols]
    subset_pos = df[df['x10'] > 1.0][cols]
    
    subsets = [subset_neg, subset_range1, subset_range2, subset_pos]
    return subsets

def split_by_brand(df):
    cols = list(df.columns)
    subset_samsung = df[df['ProductBrand_Samsung'] == 1][cols]
    subset_apple = df[df['ProductBrand_Apple'] == 1][cols]
    subset_hp = df[df['ProductBrand_HP'] == 1][cols]
    subset_sony = df[df['ProductBrand_Sony'] == 1][cols]
    subset_others = df[df['ProductBrand_Other Brands'] == 1][cols]

    subsets = [subset_samsung, subset_apple, subset_hp, subset_sony, subset_others]
    return subsets

def split_by_category(df):
    cols = list(df.columns)
    subset_smartphones = df[df['ProductCategory_Smartphones'] == 1][cols]
    subset_smartwatches = df[df['ProductCategory_Smart Watches'] == 1][cols]
    subset_tablets = df[df['ProductCategory_Tablets'] == 1][cols]
    subset_laptops = df[df['ProductCategory_Laptops'] == 1][cols]
    subset_headphones = df[df['ProductCategory_Headphones'] == 1][cols]

    subsets = [subset_smartphones, subset_smartwatches, subset_tablets, subset_laptops, subset_headphones]
    return subsets

def split_by_age(df):
    cols = list(df.columns)
    subset_neg = df[df['CustomerAge'] <= 0.0][cols]
    subset_range1= df[(df['CustomerAge'] > 0.0) & (df['CustomerAge'] <= 0.5)][cols]
    subset_range2 = df[(df['CustomerAge'] > 0.5) & (df['CustomerAge'] <= 1.0)][cols]
    subset_pos = df[df['CustomerAge'] > 1.0][cols]
    subsets = [subset_neg, subset_range1, subset_range2, subset_pos]
    return subsets

def split_by_gender(df):
    cols = list(df.columns)
    subset_1 = df[df['CustomerGender_0'] == 1 ][cols]
    subset_2 = df[df['CustomerGender_1'] == 1 ][cols]
    
    percentage = True
    if(percentage):
        male_sample1 = subset_1.sample(frac=0.9, random_state=42)
        female_sample1 = subset_2.sample(frac=0.1, random_state=0)
        male_sample2 = subset_1.sample(frac=0.1, random_state=55)
        female_sample2 = subset_2.sample(frac=0.9, random_state=1)
        subset_1 = pd.concat([male_sample1, female_sample1])
        subset_2 = pd.concat([male_sample2, female_sample2])
    subsets = [subset_1, subset_2]
    return subsets

def split_by_satisfaction(df):
    cols = list(df.columns)
    
    subset_1 = df[(df['CustomerSatisfaction'] <= 0) & (df['CustomerSatisfaction'] > -1)][cols]
    subset_2 = df[(df['CustomerSatisfaction'] <= -1) & (df['CustomerSatisfaction'] > -2)][cols]
    subset_3 = df[(df['CustomerSatisfaction'] > 0) & (df['CustomerSatisfaction'] <= 1)][cols]
    subset_4= df[(df['CustomerSatisfaction'] > 1) & (df['CustomerSatisfaction'] <= 2)][cols]

    subsets = [subset_1, subset_2, subset_3, subset_4]
    return subsets

def split_by_doors(df):
    cols = list(df.columns)

    subset_1 = df[df['doors_2'] == 1]
    subset_2 = df[df['doors_4'] == 1]
    subset_3 = df[df['doors_more'] == 1]
    
    subsets = [subset_1, subset_2, subset_3]
    return subsets

def split_by_health(df):
    cols = list(df.columns)
    
    subset_1 = df[df["'health'_priority"] == 1]
    subset_2 = df[df["'health'_recommended"] == 1]
    subset_3 = df[df["'health'_not_recom"] == 1]

    subsets = [subset_1, subset_2, subset_3]
    return subsets

def split_by_odor(df):
    cols = list(df.columns)
    
    subset_1 = df[df["Odor_foul"] == 1]
    subset_2 = df[df["Odor_none"] == 1]
    subset_3 = df[df["Odor_almond"] == 1]
    subset_4 = df[df["Odor_anise"] == 1]
    subset_5 = df[df["Odor_spicy"] == 1]
    subset_6 = df[df["Odor_pungent"] == 1]
    subset_7 = df[df["Odor_fishy"] == 1]
    subset_8 = df[df["Odor_creosote"] == 1]

    subsets = [subset_1, subset_2, subset_3, subset_4, subset_5, subset_6, subset_7, subset_8]
    return subsets

def split_by_a1(df):
    cols = df.columns

    subset_1 = df[df['A1'] <= -0.5]
    subset_2 = df[(df['A1'] > -0.5) & (df['A1'] <= 0.5)]
    subset_3 = df[df['A1'] > 0.5]

    subsets = [subset_1, subset_2, subset_3]
    return subsets

def split_by_v1(df):
    cols = df.columns
    subset_1 = df[df['V1'] <= 1.0]
    subset_2 = df[(df['V1'] > 1.0) & (df['V1'] <= 3.0)]
    subset_3 = df[df['V1'] > 3.0]
    subsets = [subset_1, subset_2, subset_3]
    return subsets

def split_by_v2(df):
    cols = df.columns
    subset_1 = df[df['V2'] <= 1.0]
    subset_2 = df[(df['V2'] > 1.0) & (df['V2'] <= 3.0)]
    subset_3 = df[df['V2'] > 3.0]
    subsets = [subset_1, subset_2, subset_3]
    return subsets