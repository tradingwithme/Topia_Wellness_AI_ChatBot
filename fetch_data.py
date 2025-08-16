import pandas as pd

def get_mental_health_data():
    """
    Load the mental health dataset from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the mental health data.
    """
    # Load the dataset
    mental_df = pd.read_csv('mentalhealth.csv')
    df_parquet = pd.read_parquet("hf://datasets/heliosbrahma/mental_health_chatbot_dataset/data/train-00000-of-00001-01391a60ef5c00d9.parquet")
    df2 = pd.DataFrame(df_parquet["text"].apply(lambda x: {"Questions":x.splitlines()[0].split(">",1)[-1].split(":",1)[-1],
"Answers":x.splitlines()[1].split(">",1)[-1].split(":",1)[-1]}).tolist())
    return pd.concat([df2, mental_df], ignore_index=True)
