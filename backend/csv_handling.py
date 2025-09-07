import pandas as pd
from os import path, getcwd
from multiprocessing import Lock

_csv_lock = Lock()

def get_mental_health_data():
    """
    Load and combine mental health datasets:
    - Local CSV (mentalhealth.csv)
    - Hugging Face parquet dataset
    - Approved responses (if available)

    Returns:
        pd.DataFrame: Combined and deduplicated dataset with standardized schema.
    """
    with _csv_lock:
        try:
            print("Loading mentalhealth.csv...")
            mental_df = pd.read_csv("mentalhealth.csv", index_col=0)
        except FileNotFoundError:
            print("Warning: 'mentalhealth.csv' not found. Starting with empty DataFrame.")
            mental_df = pd.DataFrame(columns=["Questions", "Answers"])

        try:
            print("Loading Hugging Face parquet dataset...")
            df_parquet = pd.read_parquet(
                "hf://datasets/heliosbrahma/mental_health_chatbot_dataset/data/train-00000-of-00001-01391a60ef5c00d9.parquet"
            )

            df2 = pd.DataFrame(
                df_parquet["text"]
                .apply(
lambda x: {
"Questions": x.splitlines()[0]
.split(">", 1)[-1]
.split(":", 1)[-1]
.strip(),
"Answers": x.splitlines()[1]
.split(">", 1)[-1]
.split(":", 1)[-1]
.strip(),
}
)
.tolist()
)
        except Exception as e:
            print(f"Warning: Could not load parquet dataset: {e}")
            df2 = pd.DataFrame(columns=["Questions", "Answers"])
        combined_df = pd.concat([mental_df, df2], ignore_index=True)
        combined_df = combined_df.drop_duplicates().reset_index(drop=True)
        combined_df["Type"] = "O"
        approved_path = path.join(getcwd(), "approved_responses.csv")
        if path.exists(approved_path):
            print("Loading approved responses...")
            try:
                approved_responses = pd.read_csv(approved_path, index_col=0)
                combined_df = pd.concat([combined_df, approved_responses], ignore_index=True)
                combined_df = combined_df.drop_duplicates().reset_index(drop=True)
            except Exception as e:
                print(f"Warning: Could not load approved_responses.csv: {e}")

        print(f"✅ Dataset loaded successfully with {len(combined_df)} entries.")
        return combined_df
