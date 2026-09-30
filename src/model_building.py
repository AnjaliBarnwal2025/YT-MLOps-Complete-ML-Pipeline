import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

## Ensures the log directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

## Logging Configuration
logger=logging.getLogger("model_bulding")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
logger.setLevel(logging.DEBUG)

log_file_path=os.path.join(log_dir,"model_building.log")
file_handler=logging.FileHandler(log_file_path)
logger.setLevel(logging.DEBUG)

## Forming formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    :param file path: Path to the CSV file
    :return: Loaded Dataframe
    """
    try:
        df=pd.read_csv(file_path)
        logger.debug(f"Data Loaded successfully from {file_path} of shape {df.shape}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the csv file {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured while loading the data {e}")
        raise

def train_model(X_train:np.ndarray, y_train:np.ndarray, params:dict)->RandomForestClassifier:
    """
    Train the RandomForest model
    
    :param X_train: Training Features
    :param y_train: Training Labels
    :param params: Dictionary of hyperparameters
    :return Trained RandomForest Classifier
    """
    try:
        if X_train.shape[0] !=y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_trai must be the same.")
        
        logger.debug(f"Initializing Random forest model with parameters: {params}")
        clf=RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug(f"Model training started with {X_train.shape[0]}")
        clf.fit(X_train,y_train)
        logger.debug("Model training Completed!")

        return clf
    except ValueError as e:
        logger.error(f"Value error during model training:{e}")
        raise
    except Exception as e:
        logger.error(f"Error during model training:{e}")
        raise

def save_model(model, file_path:str)->None:
    """
    Save the trained model to a file.
    :param model: Trained model object.
    :param file_path: Path to save the model file.
    """
    try:
        ## Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug(f"Model saved to: {file_path}")

    except FileNotFoundError as e:
        logger.error(f"File does not exists:{e}")
        raise
    except Exception as e:
        logger.error(f"Errir occured while saving the model:{e}")
        raise

def main():
    try:
        params={"n_estimators":25,"random_state":2}
        train_data=load_data("./data/processed/train_tfidf.csv")
        X_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values

        clf=train_model(X_train,y_train,params)
        model_save_path='models/model.pkl'
        save_model(clf,model_save_path)

    except Exception as e:
        logger.error(f"Failed to complete the model building process:{e}")
        print(f"Error: {e}")

if __name__=='__main__':
    main()