# 🩺 ChronicGuard AI  
**AI-Driven Risk Prediction for Chronic Care**

ChronicGuard AI is a prototype **clinical decision support system** designed to proactively predict the **90-day deterioration risk** for patients with chronic conditions like **diabetes, heart failure, and obesity**.  

By combining a **high-performance ML model** with an **intuitive dashboard**, ChronicGuard AI shifts healthcare from **reactive → proactive**, empowering clinicians to **intervene earlier** and improve patient outcomes.

---

## ✨ Key Features

- ⚡ **High-Performance Prediction Model**  
  Uses [LightGBM](https://lightgbm.readthedocs.io/) with **AUROC ~0.88** for patient risk prediction.

- 🧠 **Clinically-Aware Explainability (XAI)**  
  Not just scores — the dashboard highlights **key clinical drivers** (e.g., *Creatinine - Trending Upward*).

- 🧪 **Digital Twin Intervention Simulator**  
  Built with **PyTorch RNNs** to forecast health trajectories and simulate interventions (e.g., med adherence → risk reduction).

- 📊 **Interactive Dashboard**  
  - **Cohort Control Tower**: risk-stratified overview of 100+ patients  
  - **Patient Deep-Dive**: temporal graphs + risk factors  
  - **Intervention Simulator**: patient-specific outcomes  
  - **Model Analytics**: ROC, PR curves, confusion matrix  

- 🔍 **Automated Evaluation Suite**  
  Automatically generates ROC, PR, and confusion matrix plots into `/artifacts` for transparency.

---

## 🖼️ Dashboard Preview

- **Cohort Dashboard** – risk-stratified patient view  
- **Patient Deep-Dive** – temporal graphs + risk drivers  
- **Intervention Simulator** – outcomes of interventions  
- **Model Analytics** – evaluation plots auto-generated  



---

## 🛠️ Tech Stack

- **Machine Learning**: Python, LightGBM, Scikit-learn  
- **Digital Twin Simulator**: PyTorch  
- **Backend & API**: Flask  
- **Frontend**: HTML, CSS, JavaScript (Chart.js)  
- **Automation**: PowerShell  

---

## 🚀 Getting Started

### ✅ Prerequisites
- Python **3.8+**
- Windows **PowerShell**

### ⚡ Installation & Launch

```powershell
# Clone repository
git clone https://github.com/Craftz21/AIriskPrediction.git
cd AIriskPrediction

# Run setup script (creates venv, installs packages, trains models, launches server)
.\script.ps1
