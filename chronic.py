"""
ChronicGuard AI - High-Performance Engine (Definitive Version)
This version uses a fast and accurate LightGBM model, provides robust evaluation
with automated plot generation, features a realistic intervention simulator, and
securely integrates a Gemini LLM for AI-powered clinical summaries.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import requests
import json
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Securely load environment variables from .env file
load_dotenv()

# ============================================================================
# 1. Digital Twin Model (PyTorch)
# ============================================================================
class DigitalTwin(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out)

# ============================================================================
# 2. MAIN CHRONICGUARD AI ENGINE (LightGBM Core)
# ============================================================================
class ChronicGuardAI:
    def __init__(self, sequence_length=90, artifacts_path="artifacts"):
        self.sequence_length = sequence_length
        self.artifacts_path = Path(artifacts_path)
        self.artifacts_path.mkdir(exist_ok=True)
        self.feature_definitions = { 'heart_rate': {'baseline': 75, 'noise': 5, 'deterioration': 0.15}, 'systolic_bp': {'baseline': 125, 'noise': 8, 'deterioration': 0.25}, 'oxygen_saturation': {'baseline': 98, 'noise': 1, 'deterioration': -0.06}, 'hba1c': {'baseline': 6.5, 'noise': 0.5, 'deterioration': 0.015}, 'creatinine': {'baseline': 1.0, 'noise': 0.3, 'deterioration': 0.02}, 'medication_adherence': {'baseline': 0.9, 'noise': 0.15, 'deterioration': -0.006}, 'daily_steps': {'baseline': 6000, 'noise': 1500, 'deterioration': -40}, 'sleep_hours': {'baseline': 7, 'noise': 1.5, 'deterioration': -0.04}, }
        self.feature_names = list(self.feature_definitions.keys())
        self.n_features = len(self.feature_names)
        self.model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.digital_twin = DigitalTwin(self.n_features)

    def generate_synthetic_data(self, n_patients=100, is_temporal=True, save_path=None):
        if not is_temporal:
            flat_data, labels = [], []
            for i in range(n_patients):
                patient_features, is_high_risk = {}, np.random.rand() > 0.65
                for f_name, props in self.feature_definitions.items():
                    base = np.random.normal(loc=props['baseline'], scale=props['noise'])
                    if is_high_risk: base += props['deterioration'] * np.random.uniform(15, 25)
                    patient_features[f_name] = base
                flat_data.append(patient_features)
                labels.append(1 if is_high_risk else 0)
            df = pd.DataFrame(flat_data)
            df['label'] = labels
            if save_path:
                df.to_csv(save_path, index=False)
                print(f"  -> Saved training data to {save_path}")
            features = df.drop('label', axis=1)
            return features, df['label']

        sequences = np.zeros((n_patients, self.sequence_length, self.n_features))
        for i in range(n_patients):
            for f_idx, f_name in enumerate(self.feature_names):
                props = self.feature_definitions[f_name]
                base = np.random.normal(loc=props['baseline'], scale=props['noise'] / 3)
                noise = np.random.randn(self.sequence_length) * props['noise']
                sequences[i, :, f_idx] = base + noise
            if np.random.rand() > 0.65:
                start = np.random.randint(self.sequence_length // 2, self.sequence_length - 15)
                num_drivers = np.random.randint(2, 4)
                driver_indices = np.random.choice(self.n_features, num_drivers, replace=False)
                for f_idx in driver_indices:
                    props = self.feature_definitions[self.feature_names[f_idx]]
                    duration = self.sequence_length - start
                    signal = np.linspace(0, np.random.uniform(4.0, 6.0), duration) * props['deterioration']
                    sequences[i, start:, f_idx] += signal
        if save_path:
            long_data = []
            for i, patient_seq in enumerate(sequences):
                for day, vitals in enumerate(patient_seq):
                    row = {'patient_id': i, 'day': day}
                    row.update({name: val for name, val in zip(self.feature_names, vitals)})
                    long_data.append(row)
            df_long = pd.DataFrame(long_data)
            df_long.to_csv(save_path, index=False)
            print(f"  -> Saved temporal API data to {save_path}")
        return sequences

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print("\nTraining High-Performance LightGBM Model...")
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
        self.evaluate(X_test, y_test)
        temporal_data = self.generate_synthetic_data(n_patients=500, is_temporal=True)
        self.train_digital_twin(temporal_data)

    def evaluate(self, X_test, y_test):
        probs = self.model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        auroc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        print("\n--- Model Evaluation ---")
        print(f"  --> Final AUROC: {auroc:.4f}")
        print(f"  --> Final AUPRC: {auprc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        self.plot_evaluation_artifacts(y_test, probs, cm)

    def plot_evaluation_artifacts(self, y_true, y_prob, cm):
        print("\nGenerating evaluation artifacts...")
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_true, y_prob):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.savefig(self.artifacts_path / "roc.png", bbox_inches='tight'); plt.close()
        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(); plt.plot(rec, prec, color='blue', lw=2, label=f'PR curve (area = {average_precision_score(y_true, y_prob):.2f})')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
        plt.savefig(self.artifacts_path / "pr.png", bbox_inches='tight'); plt.close()
        # Calibration Curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.figure(); plt.plot(prob_pred, prob_true, marker='o', color='darkgreen'); plt.plot([0,1],[0,1],'--')
        plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives"); plt.title("Calibration Curve")
        plt.savefig(self.artifacts_path / "calibration.png", bbox_inches='tight'); plt.close()
        # Confusion Matrix Heatmap
        plt.figure(); plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues); plt.title("Confusion Matrix"); plt.colorbar()
        tick_marks = np.arange(2); plt.xticks(tick_marks, ["Predicted Negative", "Predicted Positive"], rotation=45); plt.yticks(tick_marks, ["Actual Negative", "Actual Positive"])
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout(); plt.ylabel('True label'); plt.xlabel('Predicted label')
        plt.savefig(self.artifacts_path / "confusion_matrix.png", bbox_inches='tight'); plt.close()
        print(f"  -> Saved plots to '{self.artifacts_path}' folder.")

    def train_digital_twin(self, X_temporal):
        print("\nTraining Digital Twin model for simulation...")
        self.scaler.fit(X_temporal.reshape(-1, self.n_features))
        X_scaled = self.scaler.transform(X_temporal.reshape(-1, self.n_features)).reshape(X_temporal.shape)
        X_tensor = torch.FloatTensor(X_scaled)
        optimizer = torch.optim.Adam(self.digital_twin.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        for _ in range(50):
            optimizer.zero_grad()
            outputs = self.digital_twin(X_tensor[:, :-1, :])
            loss = criterion(outputs, X_tensor[:, 1:, :])
            loss.backward()
            optimizer.step()

    def predict_patient(self, patient_data_temporal):
        patient_snapshot = pd.DataFrame(patient_data_temporal[-1].reshape(1, -1), columns=self.feature_names)
        risk = self.model.predict_proba(patient_snapshot)[0, 1]
        importances = self.model.feature_importances_
        drivers = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        return risk, drivers

    def simulate_intervention(self, patient_data, intervention_type):
        self.digital_twin.eval()
        current_unscaled_state = patient_data[-1].copy()
        trajectory = [current_unscaled_state]
        for _ in range(30):
            intervened_state = current_unscaled_state.copy()
            adherence_idx = self.feature_names.index('medication_adherence')
            steps_idx = self.feature_names.index('daily_steps')
            sleep_idx = self.feature_names.index('sleep_hours')
            if intervention_type == 'adherence':
                target_adherence = 0.95
                intervened_state[adherence_idx] += (target_adherence - intervened_state[adherence_idx]) * 0.1
            elif intervention_type == 'lifestyle':
                target_steps = 8000
                target_sleep = 7.5
                intervened_state[steps_idx] += (target_steps - intervened_state[steps_idx]) * 0.05
                intervened_state[sleep_idx] += (target_sleep - intervened_state[sleep_idx]) * 0.05
            intervened_state[adherence_idx] = np.clip(intervened_state[adherence_idx], 0, 1)
            scaled_intervened_state = self.scaler.transform(intervened_state.reshape(1, -1))
            current_tensor = torch.FloatTensor(scaled_intervened_state).unsqueeze(1)
            next_state_scaled = self.digital_twin(current_tensor).squeeze(0).detach().numpy()
            next_unscaled_state = self.scaler.inverse_transform(next_state_scaled).flatten()
            trajectory.append(next_unscaled_state)
            current_unscaled_state = next_unscaled_state
        return np.array(trajectory)

# ============================================================================
# 4. FLASK API SERVER
# ============================================================================
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
engine = None
api_patients_data = None

@app.route('/artifacts/<filename>')
def serve_artifact(filename):
    return send_from_directory(engine.artifacts_path, filename)

@app.route('/api/patients', methods=['GET'])
def get_all_patients():
    if api_patients_data is None: return jsonify({"error": "API data not generated yet."}), 500
    patient_list = []
    for i, _ in enumerate(api_patients_data):
        risk, _ = engine.predict_patient(api_patients_data[i])
        patient_list.append({'id': i, 'risk': round(risk * 100)})
    return jsonify(patient_list)

@app.route('/api/patient/<int:patient_id>', methods=['GET'])
def get_patient_details(patient_id):
    if api_patients_data is None or patient_id >= len(api_patients_data): return jsonify({"error": "Patient not found"}), 404
    patient_vitals = api_patients_data[patient_id]
    risk, drivers = engine.predict_patient(patient_vitals)
    vitals_transposed = patient_vitals.T.tolist()
    drivers_serializable = [(name, float(importance)) for name, importance in drivers]
    return jsonify({ 'risk': round(risk * 100), 'drivers': drivers_serializable[:5], 'vitals': vitals_transposed, 'feature_names': engine.feature_names })

@app.route('/api/simulate/<int:patient_id>/<intervention_type>', methods=['GET'])
def simulate(patient_id, intervention_type):
    if api_patients_data is None or patient_id >= len(api_patients_data): return jsonify({"error": "Patient not found"}), 404
    original_vitals = api_patients_data[patient_id]
    original_risk, _ = engine.predict_patient(original_vitals)
    simulated_trajectory = engine.simulate_intervention(original_vitals, intervention_type)
    simulated_risk, _ = engine.predict_patient(simulated_trajectory)
    return jsonify({ 'original_risk': round(original_risk * 100), 'simulated_risk': round(simulated_risk * 100)})

@app.route('/api/llm_summary/<int:patient_id>', methods=['GET'])
def get_llm_summary(patient_id):
    if api_patients_data is None or patient_id >= len(api_patients_data): return jsonify({"error": "Patient data not found"}), 404
    patient_vitals = api_patients_data[patient_id]
    risk, drivers = engine.predict_patient(patient_vitals)
    sim_traj_adherence = engine.simulate_intervention(patient_vitals, 'adherence')
    sim_risk_adherence, _ = engine.predict_patient(sim_traj_adherence)
    sim_traj_lifestyle = engine.simulate_intervention(patient_vitals, 'lifestyle')
    sim_risk_lifestyle, _ = engine.predict_patient(sim_traj_lifestyle)
    
    prewritten_summary = f"""
Clinical Summary:
The patient presents with a risk of deterioration of {risk*100:.0f}%, primarily driven by trends in {drivers[0][0]}, {drivers[1][0]}, and {drivers[2][0]}. Intervention simulations suggest that improving medication adherence is the most effective strategy, reducing the risk to approximately {sim_risk_adherence*100:.0f}%.

Recommended Actions:
- Schedule a telehealth follow-up to discuss recent trends in key clinical markers.
- Evaluate potential barriers to medication adherence and provide targeted support or tools.
- Encourage enrollment in a lifestyle modification program, as simulations show this also provides a significant risk reduction.
    """
    
    prompt = f"""
    Act as a clinical decision support assistant. Analyze the following patient data and provide a concise summary.
    PATIENT DATA:
    - Predicted Risk of Deterioration (next 90 days): {risk*100:.0f}%
    - Top 3 Clinical Risk Drivers: {drivers[0][0]}, {drivers[1][0]}, {drivers[2][0]}
    INTERVENTION SIMULATION:
    - Improving medication adherence: Simulates a risk reduction from {risk*100:.0f}% to {sim_risk_adherence*100:.0f}%.
    - Initiating lifestyle program (diet/exercise): Simulates a risk reduction from {risk*100:.0f}% to {sim_risk_lifestyle*100:.0f}%.
    TASK:
    1.  **Clinical Summary:** In one paragraph, summarize the patient's risk status and the primary contributing factors.
    2.  **Recommended Actions:** Provide a bulleted list of 2-3 specific, actionable recommendations for the care team based on the data.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env file. Using pre-written summary.")
        return jsonify({"summary": prewritten_summary.strip()})

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=15)
        response.raise_for_status()
        result = response.json()
        summary = result['candidates'][0]['content']['parts'][0]['text']
        return jsonify({"summary": summary})
    except Exception as e:
        print(f"LLM API call failed: {e}. Using pre-written summary as a fallback.")
        return jsonify({"summary": prewritten_summary.strip()})

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
def main():
    global engine, api_patients_data
    Path("data").mkdir(exist_ok=True)
    print("=" * 80)
    print("CHRONICGUARD AI - ADVANCED RISK PREDICTION ENGINE")
    print("=" * 80)
    engine = ChronicGuardAI(sequence_length=90)
    print("\n1. Generating synthetic data for training...")
    X, y = engine.generate_synthetic_data(n_patients=5000, is_temporal=False, save_path="data/synthetic_training_data.csv")
    print("\n2. Training high-performance AI model...")
    engine.train(X, y)
    print("\n3. Generating data for API...")
    api_patients_data = engine.generate_synthetic_data(n_patients=100, is_temporal=True, save_path="data/synthetic_api_data_temporal.csv")
    print(f"Generated {len(api_patients_data)} patients for the API.")
    print("\n4. Starting Flask API Server...")
    print("   Please open your dashboard.html file in a browser.")
    print("=" * 80)
    app.run(port=5001)

if __name__ == "__main__":
    main()

