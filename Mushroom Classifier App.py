import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

# Main function to run the app
def main():
    st.title("Mushroom Classifier App")
    st.sidebar.title("Model Selection Panel")
    st.markdown("Determine if mushrooms are **edible** or **poisonous** based on their features 🍄.")
    st.sidebar.markdown("Use this panel to select a classification model and configure its settings.")

    # Load and preprocess the data
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def split_data(df):
        y = df.type  # Target variable
        x = df.drop(columns=['type'])  # Features
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    # Plot metrics
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names, ax=ax)
            st.pyplot(fig)
            
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
            
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

    # Load data and split it into training and test sets
    df = load_data()
    x_train, x_test, y_train, y_test = split_data(df)
    class_names = ['Edible', 'Poisonous']

    # Sidebar: Model selection
    st.sidebar.subheader("Select Classifier")
    classifier = st.sidebar.selectbox("Choose a classification model:", 
                                       ("Support Vector Machine", "Logistic Regression", "Random Forest"))

    # Classifier-specific settings and execution
    if classifier == "Support Vector Machine":
        st.sidebar.subheader("Hyperparameters")
        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.01)
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"))
        gamma = st.sidebar.radio("Gamma", ("scale", "auto"))
        metrics = st.sidebar.multiselect("Metrics to plot:", 
                                          ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Run Classification"):
            st.subheader("Support Vector Machine Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Hyperparameters")
        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.01)
        max_iter = st.sidebar.slider("Max Iterations", 100, 500)
        metrics = st.sidebar.multiselect("Metrics to plot:", 
                                          ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Run Classification"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics)

    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of Trees", 100, 1000, step=10)
        max_depth = st.sidebar.number_input("Max Depth", 1, 20, step=1)
        bootstrap = st.sidebar.radio("Use Bootstrap Samples", (True, False))
        metrics = st.sidebar.multiselect("Metrics to plot:", 
                                          ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Run Classification"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics)

    # Option to display raw data
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Mushroom Dataset")
        st.write(df)

if __name__ == '__main__':
    main()
