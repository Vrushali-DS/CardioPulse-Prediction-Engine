import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def load_or_create_data():
    """
    Ensures clean data exists. If not, runs the data preparation script.
    """
    filepath = "data/clean_patients.csv"
    if not os.path.exists(filepath):
        print(f"Clean dataset not found at '{filepath}'. Running data preparation first...")
        from data_prep import generate_patient_data, clean_data
        os.makedirs("data", exist_ok=True)
        raw_df = generate_patient_data(n_samples=500, random_seed=42)
        clean_df = clean_data(raw_df)
        clean_df.to_csv(filepath, index=False)
    
    return pd.read_csv(filepath)

def evaluate_with_custom_threshold(model, X_test, y_test, threshold=0.35):
    """
    Evaluates model predictions using a custom probability decision boundary
    to prioritize RECALL (catching high-risk patients / reducing false negatives).
    """
    # Get probability of class 1 (heart disease)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Classify based on the lowered clinical decision threshold
    custom_preds = (probs >= threshold).astype(int)
    
    return custom_preds, probs

def main():
    # 1. Load Data
    df = load_or_create_data()
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # 2. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("Dataset Summary:")
    print(f"Total samples: {len(df)} (Heart Disease rate: {y.mean() * 100:.1f}%)")
    print(f"Training set: {X_train.shape[0]} | Testing set: {X_test.shape[0]}\n")
    
    # 3. 5-Fold Cross-Validation Setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- MODEL 1: Baseline Logistic Regression ---
    print("==============================================")
    print("TRAINING MODEL 1: Baseline Logistic Regression")
    print("==============================================")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # CV Accuracy
    lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {lr_cv_scores.mean() * 100:.2f}% (+/- {lr_cv_scores.std() * 100:.2f}%)")
    
    # Train final baseline model
    lr_model.fit(X_train, y_train)
    
    # Evaluate at default threshold (0.50)
    y_pred_lr_def = lr_model.predict(X_test)
    print("\nLogistic Regression Classification Report (Default Threshold = 0.50):")
    print(classification_report(y_test, y_pred_lr_def))
    
    
    # --- MODEL 2: Advanced Random Forest Classifier ---
    print("\n==============================================")
    print("TRAINING MODEL 2: Advanced Random Forest")
    print("==============================================")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        min_samples_split=5, 
        random_state=42,
        class_weight='balanced' # balances weights to support recall
    )
    
    # CV Accuracy
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {rf_cv_scores.mean() * 100:.2f}% (+/- {rf_cv_scores.std() * 100:.2f}%)")
    
    # Train final advanced model
    rf_model.fit(X_train, y_train)
    
    # Evaluate at default threshold (0.50)
    y_pred_rf_def = rf_model.predict(X_test)
    print("\nRandom Forest Classification Report (Default Threshold = 0.50):")
    print(classification_report(y_test, y_pred_rf_def))
    
    # 4. Clinically Optimize Decision Boundary for RECALL
    # In patient diagnosis, failing to identify a sick patient (False Negative) is much worse 
    # than calling a healthy patient in for extra screening (False Positive).
    # Thus, we lower the threshold to catch more potential cases.
    CLINICAL_THRESHOLD = 0.35
    
    print("\n==============================================")
    print(f"CLINICAL RECALL OPTIMIZATION (Threshold = {CLINICAL_THRESHOLD})")
    print("==============================================")
    
    for name, model in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
        preds, probs = evaluate_with_custom_threshold(model, X_test, y_test, threshold=CLINICAL_THRESHOLD)
        
        # Calculate standard metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        
        print(f"\n--- {name} @ Threshold {CLINICAL_THRESHOLD} ---")
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {prec * 100:.2f}%")
        print(f"Recall:    {rec * 100:.2f}%  <-- (Target Metric)")
        print(f"F1-Score:  {f1 * 100:.2f}%")
        print(f"Confusion Matrix:")
        print(f"   [True Normal: {tn:3d} | False Positive (Healthy but flagged): {fp:3d}]")
        print(f"   [False Negative (Missed Patients): {fn:3d} | True Heart Disease: {tp:3d}]")
        print(f"Clinical Note: Missed high-risk patients reduced from {confusion_matrix(y_test, model.predict(X_test))[1, 0]} to {fn}!")

    # 5. Feature Importance from Random Forest
    importances = rf_model.feature_importances_
    features = X.columns
    feat_imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)
    print("\nRandom Forest Feature Importances:")
    print(feat_imp_df.to_string(index=False))

    # 6. Save the Champion Model (Random Forest with weights and metadata)
    os.makedirs("models", exist_ok=True)
    model_payload = {
        "model": rf_model,
        "feature_names": list(X.columns),
        "logistic_regression": lr_model,
        "clinical_threshold": CLINICAL_THRESHOLD,
        "feature_importances": feat_imp_df.to_dict(orient="records"),
        "metrics_summary": {
            "rf_cv_acc": rf_cv_scores.mean(),
            "lr_cv_acc": lr_cv_scores.mean()
        }
    }
    
    model_path = "models/heart_disease_classifier.joblib"
    joblib.dump(model_payload, model_path)
    print(f"\nSaved clinical model package to '{model_path}' successfully!")

if __name__ == "__main__":
    main()
