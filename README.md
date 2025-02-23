# Body Fat Prediction Web App

This is a simple web application built using the Flask framework that predicts the body fat percentage of a person based on their physical attributes like age, height, weight, etc. The app uses a trained model to predict the body fat percentage based on user inputs.

## Example

**Input:**

<img src="https://github.com/dhavaldalvi/body-fat-prediction-web-application/blob/main/screenshots/home.JPG" width="45%" height="45%" />

**Prediction:**

![Screenshot of result page](https://github.com/dhavaldalvi/body-fat-prediction-web-application/blob/main/screenshots/result.JPG)

---

## Features

- User-friendly web interface to input physical parameters (age, height, weight, etc.).
- Body fat percentage prediction based on user input.
- Simple and lightweight interface with immediate results.

## Requirements

To run this project, you'll need to have the following installed:

- Python 3.12
- Flask
- Scikit-learn
- Pandas (for data handling)
- Numpy (for numerical operations)

## Installation

### Prerequisites

Before running the application, make sure you have the following installed:

- **Python 3.12** 
- **pip** (Python's package installer)

### Or

- **Anaconda** or **Miniconda** (for managing Python environments)

### 1. Create the Environment

Using `venv` of **Python**. Run the following command in **Command Prompt** or **Terminal**.

```bash
python -m venv myenv
```
### Or

Using `conda` of **Anaconda**. Run following command in **Anaconda Prompt**.

```bash
conda create --name myenv
```
`myenv` is the name of the directory where the virtual environment will be created. You can replace `myenv` with any name you prefer.

### 2. Activating Environment

If using **Python**

In **Windows**
```
.\myenv\Scripts\activate
```
In **MacOS/Linux**
```
source myenv/bin/activate
```
Or if using **Anaconda**

In **Windows**
```
conda activate myenv
```
In **MacOS/Linux**
```
source activate myenv
```

### 3. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/dhavaldalvi/body-fat-prediction-web-application.git
```

### 4. Install Dependencies

Navigate to the project directory and install the required packages using pip if using **Python**

```bash
cd body-fat-prediction-web-application
pip install -r requirements.txt
```
or using conda if using **Anaconda** 

```
cd body-fat-prediction-web-application
conda install --yes --file requirements.txt
```

The `requirements.txt` file contain the necessary libraries.

### 5. Run the Flask App

To start the application, run the following command:

```bash
python app.py
```

This will start the Flask server and generat a pickle file with extension '*.pkl'. By default, the app will be hosted at `http://127.0.0.1:5000/`.

---

## Usage

Once the app is running, open your browser and navigate to `http://127.0.0.1:5000/`. You will see a form where you can input the following parameters:

- Age (years)
- Weight (Kg)
- Height (cm)
- Neck circumference (cm)
- Chest circumference (cm)
- Abdomen 2 circumference (cm)
- Hip circumference (cm)
- Thigh circumference (cm)
- Knee circumference (cm)
- Ankle circumference (cm)
- Biceps (extended) circumference (cm)
- Forearm circumference (cm)
- Wrist circumference (cm)

After entering the values, click on the "Submit" button, and the model will predict the percentage of body fat.

The result will be displayed immediately, and it will indicate the predicted percentage of body fat.

---

## Model Details

- **Dataset**: The model was trained using a publicly available Body Fat Prediction dataset (https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset)
- **Machine Learning Algorithm**: A machine learning algorithm Lasso Regression was used to train the model.
- **Model File**: The trained model is saved as a `.pkl` file.




