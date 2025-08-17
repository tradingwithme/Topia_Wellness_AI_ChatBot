import pandas as pd
from os import path, getcwd

def get_mental_health_data():
    """
    Load the mental health dataset from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the mental health data.
    """
    # Load the dataset
    mental_df = pd.read_csv('mentalhealth.csv',index_col=0)
    df_parquet = pd.read_parquet("hf://datasets/heliosbrahma/mental_health_chatbot_dataset/data/train-00000-of-00001-01391a60ef5c00d9.parquet")
    df2 = pd.DataFrame(df_parquet["text"].apply(lambda x: {"Questions":x.splitlines()[0].split(">",1)[-1].split(":",1)[-1],
"Answers":x.splitlines()[1].split(">",1)[-1].split(":",1)[-1]}).tolist())
    combined_df = pd.concat([df2, mental_df], ignore_index=True).drop_duplicates()
    combined_df.loc[:,"Type"] = "O"
    if path.exists(path.join(getcwd(),"approved_responses.csv")):
        mental_health_results = pd.read_csv("approved_responses.csv",index_col=0)
        combined_df = pd.concat([combined_df, mental_health_results], ignore_index=True)
    return combined_df
