# 🔍 Unsupervised Anomaly Detection on Tabular Data

This project is a **general-purpose anomaly detection tool** using **unsupervised machine learning**, designed to work on any CSV dataset. Whether it’s detecting fraud in financial transactions or identifying abnormal behavior in refinery equipment, this tool lets you **clean data, apply Isolation Forest, and detect anomalies — all without needing labeled data**.

> ✅ Bonus: Includes a **Streamlit GUI** that allows users to upload their own datasets and detect anomalies interactively.

---

## 🧠 Project Overview

- 📌 **Goal**: Automatically detect anomalies (outliers) in tabular data where labels like “Normal” or “Faulty” are **not available**  
- 🧠 **ML Algorithm**: Isolation Forest (Unsupervised Learning)  
- 🛠️ **Use Cases**:  
  - Industrial asset monitoring (refinery sensor logs)  
  - Credit card fraud detection  
  - Network/server failure prediction  
  - IoT device anomaly tracking  
- ⚙️ **Tech Stack**: Python, Pandas, Scikit-learn, Seaborn, Streamlit, Jupyter Notebook

---

## 🗂️ Folder Structure
<pre>unsupervised-anomaly-detector/
│
├── data/ # Example datasets (CSV)
├── Unsupervised_Detection.ipynb # Notebook: load, clean, model
├── streamlit_app/
│ └── app.py # Streamlit GUI
├── requirements.txt # Dependencies
└── README.md # Project documentation</pre>


---

## 💡 How It Works

1. **User Input**: Upload any tabular `.csv` dataset (e.g., equipment logs, transaction records, sensor data).
2. **Preprocessing**: Cleans missing values, scales numerical data.
3. **Anomaly Detection**: Applies Isolation Forest to detect anomalies based on data behavior.
4. **Output**: Labels rows as `"Normal"` or `"Anomaly"` and optionally exports the result.

---

## 📦 Technologies Used

| Tool         | Role                                  |
|--------------|----------------------------------------|
| Python       | Core programming language              |
| Pandas       | Data manipulation                      |
| Scikit-learn | ML algorithm (Isolation Forest)        |
| Seaborn      | Visualizations                         |
| Streamlit    | GUI for interactive anomaly detection  |
| Jupyter      | Prototyping and step-by-step learning  |

---

## 🌐 Streamlit Web App

> This app allows **non-technical users** to upload any CSV file and run anomaly detection interactively.

### 🔧 Features:

- Upload any `.csv` file
- Auto-cleaning of missing/null values
- Scaling of numeric features
- Anomaly detection using Isolation Forest
- Visual summary of results
- Downloadable output with anomaly labels

---

### 🚀 To Launch the App:

```bash
cd streamlit_app
streamlit run app.py
```


## ⚙️ Setup Instructions
1. Clone the Repository
```bash
git clone https://github.com/ArshSharan/Unsupervised-Anomaly-Detection.git
cd unsupervised-anomaly-detector
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run the Notebook
```bash
jupyter notebook
# Open `Unsupervised_Detection.ipynb`
```

4. Launch Streamlit App
```bash
cd streamlit_app
streamlit run app.py
```
## 🔧 Real-World Use Cases

| Domain           | Example Use                                          |
| ---------------- | ---------------------------------------------------- |
| 🔧 Oil & Gas     | Detect abnormal equipment behavior using sensor logs |
| 💳 Finance       | Spot fraudulent transactions                         |
| 📶 IoT Devices   | Monitor for unexpected spikes                        |
| 🏭 Manufacturing | Detect process anomalies                             |


## 🔍 Future Additions
- [ ] Add One-Class SVM and Autoencoders for comparison

- [ ] Feature importance and SHAP visualizations

- [ ] Live data ingestion with MQTT or REST APIs

- [ ] Deployment on cloud (Heroku, Streamlit Cloud, etc.)


## 🙋‍♂️ Who Is This For?

- Data science learners.

- ML engineers building tools.

- Interns working on anomaly detection projects.


## ⭐ Like This Project?
Give it a ⭐ if it helped you or inspired you to build something cool!

---

