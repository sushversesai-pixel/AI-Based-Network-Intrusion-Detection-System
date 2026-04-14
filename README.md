# 🔒 AI-Based Network Intrusion Detection System (NIDS)

An intelligent machine learning-powered security application designed to detect and classify malicious network activity in real-time. Using advanced algorithms and interactive visualization, this system helps organizations identify and respond to security threats effectively.

## 📋 Introduction

The AI-Based Network Intrusion Detection System (NIDS) leverages machine learning to analyze network traffic and identify potential security threats. The system uses the **Random Forest algorithm** to classify network traffic as either **Benign** or **Malicious**, providing security teams with actionable intelligence through an interactive web-based dashboard powered by Streamlit.

## 🎯 Objectives

- ✅ Detect network intrusions using Machine Learning algorithms
- ✅ Classify network traffic into benign and attack categories
- ✅ Provide real-time traffic simulation and prediction capabilities
- ✅ Visualize performance metrics (accuracy, precision, recall, F1-score)
- ✅ Display confusion matrices and ROC curves for model evaluation
- ✅ Enable security teams to respond to threats proactively

## 🛠 Tech Stack

- **Language**: Python 3.8+
- **Machine Learning**: 
  - Scikit-learn (Random Forest Classifier)
  - Pandas (Data manipulation)
  - NumPy (Numerical computing)
- **Web Framework**: Streamlit (Interactive dashboard)
- **Data Visualization**: 
  - Matplotlib
  - Seaborn
  - Plotly
- **Version Control**: Git & GitHub

## 📁 Project Structure
```
AI-Based-Network-Intrusion-Detection-System/ 
├── nids_main.py # Main application file 
├── model.pkl # Trained Random Forest model 
├── requirements.txt # Python dependencies 
├── data/ # Dataset directory │ 
├── training_data.csv # Training dataset 
  │ └── test_data.csv # Test dataset 
├── utils/ # Utility functions 
  │ ├── preprocessing.py # Data preprocessing 
  │ └── visualization.py # Visualization functions 
├── README.md # Project documentation 
├── .gitignore # Git ignore rules 
└── LICENSE # License information
```
Code

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher installed on your system
- pip (Python package manager)

### Step 1: Install Python

Download Python 3.8+ from [python.org](https://www.python.org)

**Important**: During installation, check the box "Add Python to PATH"

### Step 2: Clone the Repository

```bash
git clone https://github.com/sushversesai-pixel/AI-Based-Network-Intrusion-Detection-System.git
cd AI-Based-Network-Intrusion-Detection-System
Step 3: Create Virtual Environment (Recommended)
bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Step 4: Install Dependencies
bash
pip install -r requirements.txt
Step 5: Required Libraries
The requirements.txt includes:
```
Code
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
plotly
▶️ Running the Application
Ensure virtual environment is activated (if used)

Start the Streamlit application:

bash
streamlit run nids_main.py
Access the application: The application will automatically open in your default browser at:

Code
http://localhost:8501
✨ Features
1. Synthetic Traffic Simulation
Generate realistic network traffic patterns
Create benign and malicious traffic scenarios
Adjust traffic parameters for testing
2. Machine Learning Detection
Real-time intrusion detection using Random Forest
High accuracy classification (>95%)
Handles multi-class attack scenarios
3. Model Training via UI
Train models directly from the dashboard
Monitor training progress
Experiment with different parameters
4. Performance Metrics
Accuracy, Precision, Recall, F1-Score
Confusion Matrix visualization
ROC Curve analysis
Classification Reports
5. Live Prediction
Input custom network traffic data
Get instant classification results
Confidence scores for predictions
6. Data Visualization
Interactive charts and graphs
Feature importance analysis
Traffic pattern visualization
🤖 Machine Learning Model
Algorithm: Random Forest Classifier
Why Random Forest?

Excellent for binary classification
Handles non-linear relationships
Robust to overfitting
Fast prediction time
Dataset
Source: CIC-IDS2017-style simulated traffic
Features: Network traffic characteristics (packet size, duration, protocol, etc.)
Classes:
0 → Benign Traffic (Normal)
1 → Malicious Traffic (Attack)
Model Performance
Accuracy: ~97%
Precision: ~96%
Recall: ~98%
F1-Score: ~97%
📊 User Guide
Dashboard Overview
Home Page: Overview of the system and current threat status
Traffic Simulation: Generate and simulate network traffic
Model Training: Train/retrain the ML model
Prediction: Input traffic data for intrusion detection
Analytics: View detailed performance metrics
How to Use
Simulate Traffic:

Go to "Traffic Simulation" tab
Adjust parameters (number of packets, protocols, etc.)
Click "Generate Traffic"
Train Model:

Go to "Model Training" tab
Select training dataset
Click "Train Model"
Monitor training metrics
Detect Intrusions:

Go to "Prediction" tab
Input network traffic parameters
Click "Predict"
View prediction results and confidence
View Analytics:

Go to "Analytics" tab
Review confusion matrix and ROC curve
Analyze feature importance
🔧 Customization
Modify Model Parameters
Edit nids_main.py to adjust:

Python
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Tree depth
    random_state=42
)
Add New Features
Extend the feature extraction in utils/preprocessing.py:

Python
def extract_features(packet):
    features = {
        'packet_size': len(packet),
        'protocol': packet.protocol,
        # Add custom features here
    }
    return features
📈 Performance Monitoring
Monitor system performance through:

Real-time accuracy metrics
Precision and recall analysis
Confusion matrix visualization
ROC curve for threshold optimization
🔐 Security Considerations
Never expose sensitive data in logs
Use secure connections (HTTPS) when deploying
Validate and sanitize all user inputs
Keep Python and libraries updated
Implement rate limiting for API endpoints
🚀 Deployment
Deploy on Streamlit Cloud
Push code to GitHub
Go to streamlit.io
Connect your GitHub repository
Deploy with one click
Deploy on AWS/GCP
bash
# Build Docker image
docker build -t nids .

# Run container
docker run -p 8501:8501 nids
🤝 Contributing
We welcome contributions! Please:

Fork the repository
Create a feature branch: git checkout -b feature/your-feature
Make your changes and test thoroughly
Commit: git commit -m "Add your feature"
Push: git push origin feature/your-feature
Submit a pull request
📚 Resources & References
Python Documentation
Streamlit Documentation
Scikit-learn Documentation
CIC-IDS2017 Dataset
Random Forest Algorithm
📝 License
This project is licensed under the MIT License - see LICENSE file for details

👤 Author
Sai Susmitha B
