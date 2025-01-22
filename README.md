# Mushroom Edibility Classifier  

This web application helps users classify mushrooms as **edible** or **poisonous** based on various features. The app is built using **Streamlit** and leverages popular machine learning algorithms for binary classification.  

## Features  

- Interactive interface to upload and visualize the mushroom dataset.  
- Binary classification using three models:  
  - Support Vector Machine (SVM)  
  - Logistic Regression  
  - Random Forest  
- Hyperparameter tuning for each classifier.  
- Visualizations for model evaluation, including:  
  - Confusion Matrix  
  - ROC Curve  
  - Precision-Recall Curve  
- Option to view raw data used for classification.  

## Demo  

![App Screenshot](link-to-screenshot)  

## Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/naoufalcb/Mushroom-Classifier-App.git  
   cd Mushroom-Classifier-App 
   ```  

2. Install required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Run the application:  
   ```bash  
   streamlit run Mushroom-Classifier-App.py  
   ```  

4. Open the app in your browser:  
   [http://localhost:8501](http://localhost:8501)  

## Dataset  

The application uses the [Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom) from the UCI Machine Learning Repository. This dataset contains 22 features to classify mushrooms as edible or poisonous.  

## Usage  

1. Launch the app.  
2. Select a classification model (SVM, Logistic Regression, or Random Forest) from the sidebar.  
3. Adjust the hyperparameters for the chosen model.  
4. Click on the "Classify" button to see the results, including accuracy, precision, and recall.  
5. Visualize model performance using the provided plots.  

## Screenshots  

### Main Interface  
![Main Interface](link-to-main-interface-screenshot)  

### Model Selection  
![Model Selection](link-to-model-selection-screenshot)  

### Performance Metrics  
![Performance Metrics](link-to-performance-metrics-screenshot)  

## Dependencies  

- Python 3.7+  
- Streamlit  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  

## Contributing  

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.  

## License  

This project is licensed under the [MIT License](LICENSE).  

## Contact  

For questions or feedback, feel free to reach out:  

- **Author**: Naoufal Chabaa  
- **Email**: nchabaa3@gmail.com  
