import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

## Ensure log directory exists
log_dirs='logs'
os.makedirs(log_dirs, exist_ok=True)

## Logging configuration
logger=logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path=os.path.join(log_dirs, 'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url:str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug(f"Data loaded successfully with shape {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        logger.error("No data found at the specified URL.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured while loading the data: {e}")
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 4','Unnamed: 3','Unnamed: 2'], inplace=True)
        df.rename(columns={'v1':'target','v2':'text'}, inplace=True)
        logger.debug(f"Data preprocessing completed")
        return df
    except KeyError as e:
        logger.error(f"Missing column in the dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured while preprocessing the data: {e}")
        raise

def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str)->None:
    try:
        raw_data_path=os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug(f"Train and test data saved at {raw_data_path}")
    except Exception as e:
        logger.error(f"Unexpected error occured while saving the data: {e}")
        raise

def main():
    try:
        test_size=0.2
        data_path='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv'
        df=load_data(data_url=data_path)
        final_df=preprocess_data(df)
        train_data, test_data=train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error(f'Failed to complete the data ingestion process: {e}')
        print(f"Error: {e}")

if __name__=="__main__":
    main()