import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    return pd.read_csv(file_path).dropna()


def preprocess_data(df):
    custom_mapping = {'Hazardous': 0, 'Poor': 1, 'Moderate': 2, 'Good': 3}
    df['Air Quality'] = df['Air Quality'].map(custom_mapping)

    y = df['Air Quality']
    X = df.drop('Air Quality', axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def stratified_sample(X, y, sample_size=1500, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=random_state)
    for train_index, _ in sss.split(X, y):
        X_sampled, y_sampled = X[train_index], y.iloc[train_index]
    return X_sampled, y_sampled


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_processed_data(file_path='pollution_dataset.csv', sample_size=1500):
    df = load_data(file_path)
    X, y = preprocess_data(df)

    X_sampled, y_sampled = stratified_sample(X, y, sample_size=sample_size)

    X_train, X_val, y_train, y_val = split_data(X_sampled, y_sampled)

    return X_train, y_train.values, X_val, y_val.values
