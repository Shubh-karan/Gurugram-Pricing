# 🏠 Housing Price Prediction

This project is an end-to-end workflow for predicting housing prices using the **California Housing dataset**.  
It includes data preprocessing, model training, and inference, with everything wrapped inside a single script.

---

## ⚙️ Features

- **Data Preprocessing**  
  - Handles missing values with `SimpleImputer` (median strategy).  
  - Scales numerical features using `StandardScaler`.  
  - Encodes categorical features with `OneHotEncoder`.  
  - Combines everything into a single pipeline (`ColumnTransformer`).  

- **Model Training**  
  - Uses **Stratified Shuffle Split** to create balanced train/test sets.  
  - Trains a `RandomForestRegressor` on prepared data.  
  - Saves the model (`model.pkl`) and preprocessing pipeline (`pipeline.pkl`).  

- **Inference**  
  - Loads the trained model & pipeline.  
  - Takes input from `input.csv`.  
  - Produces predictions and saves them into `output.csv`.  

---

## 📂 Project Structure


| File / Folder   | Description |
|-----------------|-------------|
| `new_main.py`   | Main script (training + inference) |
| `housing.csv`   | Dataset (required for training) |
| `input.csv`     | Test input generated during training |
| `output.csv`    | Predictions after inference |
| `model.pkl`     | Trained RandomForest model (auto-generated) |
| `pipeline.pkl`  | Preprocessing pipeline (auto-generated) |
| `.gitignore`    | Excludes large/generated files from GitHub |

## 🚀 How to Run

### 1. Train the Model
If no model exists, running the script will:
- Train a new model
- Save it as `model.pkl`
- Save preprocessing steps as `pipeline.pkl`
- Generate an `input.csv` for testing

```bash
python new_main.py

```
## 🚀 Run Inference

Once the model and pipeline are already trained (`model.pkl` and `pipeline.pkl` exist):

- Place your test data in **`input.csv`**  
- Run the script:

```bash
python new_main.py
```
