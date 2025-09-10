import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model(df, features):
    """Train the model and return it without showing plots"""
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def show_model_performance(model, df, features):
    """Show model performance metrics and plots"""
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    
    # Print metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figures for plots
    fig1, ax1 = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix')
    
    # Feature importance plot
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': features, 
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(6,3))
    sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax2)
    ax2.set_title('Feature Importances')
    
    return accuracy, fig1, fig2