
# 🚀 Customer Churn Prediction

Predict customer churn with a robust machine learning pipeline and an interactive web interface.

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── artifacts/           # Models, preprocessor, train/test CSVs
├── logs/                # Log files
├── src/
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformer.py
│   │   └── model_trainer.py
│   └── pipeline/
│       └── predict_pipeline.py
├── app.py               # Flask web app
├── jupyter.ipynb        # EDA & experimentation
├── churn.pkl            # Trained model
├── preprocessor.pkl     # Preprocessing pipeline
├── requirements.txt
├── setup.py
├── README.md
```

---

## 📝 Key Components

- **artifacts/**: Trained models, preprocessors, and data splits.
- **logs/**: Execution and error logs.
- **src/**
  - **exception.py**: Custom error handling.
  - **logger.py**: Logging utilities.
  - **utils.py**: Helper functions.
  - **components/**: Modular pipeline stages:
    - `data_ingestion.py`: Data loading & splitting
    - `data_transformer.py`: Feature engineering & preprocessing
    - `model_trainer.py`: Model training & evaluation
  - **pipeline/**: 
    - `predict_pipeline.py`: Model inference logic
- **app.py**: Flask web app for live predictions.
- **jupyter.ipynb**: Data exploration & visualization.
- **churn.pkl / preprocessor.pkl**: Serialized model & preprocessor.
- **screenshots/**: Images for documentation.

---

## ⚙️ How It Works

1. **Ingestion:** Load and split data.
2. **Transformation:** Clean and preprocess features.
3. **Training:** Train and save the best model.
4. **Prediction:** Serve predictions via Flask web app.

---

## 🌐 Web App Preview

![Screenshot 2025-04-24 110444](https://github.com/user-attachments/assets/d61bf7e6-5aa2-4f71-aef1-7dfd56150cd2)

  ![Screenshot 2025-04-24 110453](https://github.com/user-attachments/assets/f8528e96-d93b-4d57-a04a-75b7aea9d77c)

  


---

## 🚦 Quick Start

```bash
git clone https://github.com/ankit1831/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
python app.py
```
Visit [http://localhost:8080](http://localhost:8080) in your browser.

---

## 🤝 Contributing

Pull requests and suggestions are welcome!

---
