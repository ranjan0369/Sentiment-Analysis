from data_preprocess import preprocess_data
from feature_extractor import extract_features
from train_evaluate_model import train_and_evaluate

def main():
    # Step 1: Preprocess
    df = preprocess_data("datasets/imdb_dataset.csv")

    # Step 2: Feature Extraction
    X, y, vectorizer = extract_features(df)

    # Step 3: Train & Evaluate
    model, acc = train_and_evaluate(X, y, vectorizer)

if __name__ == "__main__":
    main()
