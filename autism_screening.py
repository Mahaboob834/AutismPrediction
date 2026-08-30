import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_and_check():
    # 1. Load the dataset
    try:
        df = pd.read_csv('autism_data.csv')
    except FileNotFoundError:
        print("Error: 'autism_data.csv' not found.")
        print("In Colab, upload it first using:")
        print("    from google.colab import files")
        print("    files.upload()")
        return

    # 2. Data Cleaning & Preprocessing
    df['age'] = df['age'].fillna(df['age'].median())
    df['gender'] = df['gender'].map({'f': 0, 'm': 1})
    df['jundice'] = df['jundice'].map({'no': 0, 'yes': 1})
    df['austim'] = df['austim'].map({'no': 0, 'yes': 1})
    df['Class/ASD'] = df['Class/ASD'].map({'NO': 0, 'YES': 1})

    # 3. Feature Selection
    features = [f'A{i}_Score' for i in range(1, 11)] + ['age', 'gender', 'jundice', 'austim']
    X = df[features]
    y = df['Class/ASD']

    # 4. Train/Test Split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 5. Model Training
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(scaler.transform(X_test), y_test)
    print(f"Model Training Complete. (Test Accuracy: {accuracy:.2%})")

    # 6. Interactive Prediction Section
    questions = [
        "Does the person make eye contact?",
        "Do they point to indicate interest?",
        "Do they respond to their name?",
        "Do they share interests with others?",
        "Do they play 'make-believe'?",
        "Do they follow where you look?",
        "Do they comfort others who are sad?",
        "Were their first words delayed?",
        "Do they use simple gestures?",
        "Do they stare at nothing for long periods?"
    ]

    print("\n" + "=" * 30)
    print(" AUTISM SCREENING TOOL (Adult)")
    print("=" * 30)
    print("Please answer with 1 for 'Yes' and 0 for 'No'\n")

    try:
        answers = []
        for i, q in enumerate(questions):
            val = int(input(f"Q{i + 1}: {q} (1/0): "))
            answers.append(val)

        age = float(input("Age: "))
        gender = int(input("Gender (1=Male, 0=Female): "))
        jaundice = int(input("Born with Jaundice? (1=Yes, 0=No): "))
        family_asd = int(input("Family history of ASD? (1=Yes, 0=No): "))

        user_data = answers + [age, gender, jaundice, family_asd]
        # Convert user_data to DataFrame with feature names before scaling
        user_df = pd.DataFrame([user_data], columns=features)
        user_scaled = scaler.transform(user_df)

        # 7. Results
        prediction = model.predict(user_scaled)[0]
        probability = model.predict_proba(user_scaled)[0][1]

        print("\n" + "=" * 50)
        print("======== RESULT ========".center(50))
        result_message = f"{'Potential ASD Traits Detected' if prediction == 1 else 'No ASD Traits Detected'}"
        print(result_message.center(50))
        print(f"Confidence: {probability * 100:.1f}%".center(50))
        print(f"Risk Category: {'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'}".center(50))
        print("=" * 50)
        print("Disclaimer: This is a machine learning tool, not a clinical diagnosis.")

    except ValueError:
        print("\n" + "=" * 50)
        print("Error: Please enter numeric values (1 or 0) as requested.".center(50))
        print("=" * 50)


if __name__ == "__main__":
    train_and_check()