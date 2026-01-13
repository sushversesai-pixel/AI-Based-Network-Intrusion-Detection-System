# AI-Based Network Intrusion Detection System

## 📌 Introduction
The AI-Based Network Intrusion Detection System (NIDS) is a machine learning–powered
security application designed to detect malicious network activity.
This project uses the **Random Forest algorithm** to classify network traffic as
**Benign** or **Malicious** and provides an interactive **web-based dashboard**
using Streamlit.

---

## 🎯 Objectives
- Detect network intrusions using Machine Learning
- Classify traffic into normal and attack categories
- Provide real-time traffic simulation and prediction
- Visualize performance metrics such as accuracy and confusion matrix

---

## 🧠 Technologies Used
- **Python 3.8+**
- **Machine Learning:** Random Forest (Scikit-learn)
- **Web Framework:** Streamlit
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Version Control:** Git & GitHub

---

## 🏗️ Project Structure
│
├── nids_main.py # Main application file
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files


---

## ⚙️ Installation & Setup

### 1️⃣ Install Python
Download Python 3.8 or higher from:
https://www.python.org  
Ensure **“Add Python to PATH”** is selected during installation.

### 2️⃣ Install Required Libraries
```bash
python -m pip install -r requirements.txt
▶️ How to Run the Project

Navigate to the project directory and run:

python -m streamlit run nids_main.py


The application will open automatically in your browser at:

http://localhost:8501

🧪 Features

Synthetic network traffic simulation

Machine learning–based intrusion detection

Model training through UI

Performance metrics (Accuracy & Confusion Matrix)

Live traffic input and prediction

📊 Machine Learning Model

Algorithm: Random Forest Classifier

Dataset: Simulated CIC-IDS2017-style traffic

Classification:

0 → Benign Traffic

1 → Malicious Traffic

🔒 Disclaimer

This project is developed strictly for educational and academic purposes.
It does not perform real-time packet sniffing or active intrusion.

👤 Author

Sai Susmitha

📚 References

Python Documentation – https://www.python.org

Streamlit Documentation – https://docs.streamlit.io

Scikit-learn – https://scikit-learn.org

CIC-IDS2017 Dataset – Canadian Institute for Cybersecurity


---

## 🔹 STEP 8: Push README to GitHub

After saving `README.md`:

```bash
git add README.md
git commit -m "Added project README"
git push
