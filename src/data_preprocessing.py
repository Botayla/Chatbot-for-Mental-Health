import pandas as pd

def preprocess_faq(input_path: str, output_path: str):
    """
    Preprocess the Mental Health FAQ dataset.
    
    Steps:
    1. Load CSV file.
    2. Drop the 'Question_ID' column.
    3. Combine question and answer into one text string.
    4. Save processed dataset as CSV.
    
    Args:
        input_path (str): Path to raw CSV file.
        output_path (str): Path to save processed CSV file.
    """
    # Load dataset
    df = pd.read_csv(input_path)

    # Drop unnecessary column
    if "Question_ID" in df.columns:
        df.drop(columns=["Question_ID"], inplace=True)

    # Format as Q: ... \nA: ...
    df["text"] = df.apply(lambda row: f"Q: {row['Questions']}\nA: {row['Answers']}", axis=1)

    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"[INFO] Processed dataset saved at {output_path}")

if __name__ == "__main__":
    preprocess_faq(
        input_path=r"data\Mental_Health_FAQ.csv",
        output_path=r"data\processed_Mental_Health_FAQ.csv"
    )
