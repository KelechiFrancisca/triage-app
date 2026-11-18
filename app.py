import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr

# Load Dataset
dataset = pd.read_csv('triage_dataset.csv')
print("shape:", dataset.shape)

# Map Disease to Department
dataset['Department'] = 'General Medicine'
dataset.loc[dataset['Disease'] == 'Heart attack', 'Department'] = 'Cardiology'
dataset.loc[dataset['Disease'] == 'Hypertension', 'Department'] = 'Cardiology'
dataset.loc[dataset['Disease'] == 'Common Cold', 'Department'] = 'Respiratory'
dataset.loc[dataset['Disease'] == 'Pneumonia', 'Department'] = 'Respiratory'
dataset.loc[dataset['Disease'] == 'Bronchial Asthma', 'Department'] = 'Respiratory'
dataset.loc[dataset['Disease'] == 'GERD', 'Department'] = 'Gastroenterology'
dataset.loc[dataset['Disease'] == 'Peptic ulcer disease', 'Department'] = 'Gastroenterology'
dataset.loc[dataset['Disease'] == 'Gastroenteritis', 'Department'] = 'Gastroenterology'

# Create Urgency column
dataset['Urgency'] = 'GP - Within 24hrs'
dataset.loc[dataset['Disease'] == 'Heart attack', 'Urgency'] = 'A&E - Go Now'
dataset.loc[dataset['Disease'] == 'Hypertension', 'Urgency'] = 'A&E - Go Now'
dataset.loc[dataset['Disease'] == 'Common Cold', 'Urgency'] = 'Self-Care + GP if worse'

# Fill missing symptoms
dataset = dataset.fillna('none')

# Encoding
le = LabelEncoder()
for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Department', 'Urgency']:
    dataset[col] = le.fit_transform(dataset[col].astype(str))

# Features and target
X = dataset[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7']]
y = dataset['Department']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model - Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Department names - in LabelEncoder order for Department
# Re-fit encoder on Department names to get correct order
dept_le = LabelEncoder()
dept_le.fit(['Cardiology', 'Dermatology', 'Endocrinology', 'Gastroenterology', 'General Medicine', 'General Surgery', 'Neurology', 'Orthopedics', 'Respiratory', 'Urology'])
department_names = list(dept_le.classes_)

# App logic
symptom_list = ['No More Symptom', 'Itching', 'Skin Rash', 'Continuous Sneezing', 'Shivering', 'Chills', 'Joint Pain', 'Stomach Pain', 'Acidity', 'Vomiting', 'Fatigue', 'Cough', 'High Fever', 'Headache', 'Chest Pain', 'Dizziness', 'Breathlessness', 'Sweating', 'Nausea', 'Diarrhoea', 'Constipation', 'Muscle Pain', 'Sore Throat', 'Runny Nose']

def symptom_to_number(s):
    if s in symptom_list:
        return symptom_list.index(s)
    return 0

def triage_beautiful(s1, s2, s3, s4, s5):
    symptoms = [s1, s2, s3, s4, s5]

    # SAFETY RULES
    if 'Chest Pain' in symptoms and ('Breathlessness' in symptoms or 'Sweating' in symptoms or 'Dizziness' in symptoms):
        return 'Cardiology', '🔴 A&E - GO NOW - Do not wait', 'Call emergency now. Patient needs immediate heart check. Do not let patient wait.'
    if 'High Fever' in symptoms and 'Breathlessness' in symptoms:
        return 'Respiratory', '🟠 GP - URGENT - See today', 'See doctor today. Monitor breathing and fever.'
    if 'Headache' in symptoms and 'Dizziness' in symptoms and 'Vomiting' in symptoms:
        return 'Neurology', '🟠 GP - URGENT - See today', 'See doctor today. Possible neurological issue.'

    n1 = symptom_to_number(s1)
    n2 = symptom_to_number(s2)
    n3 = symptom_to_number(s3)
    n4 = symptom_to_number(s4)
    n5 = symptom_to_number(s5)
    pred = model.predict([[n1, n2, n3, n4, n5, 0, 0]])[0]
    dept = department_names[pred]

    if dept == 'Cardiology':
        urgency = '🔴 A&E - GO NOW - Do not wait'
        advice = 'Call emergency. Patient needs immediate heart check.'
    elif dept in ['Respiratory', 'Neurology']:
        urgency = '🟠 GP - URGENT - See today'
        advice = 'See doctor today. Monitor breathing / headache.'
    elif dept == 'Gastroenterology':
        urgency = '🟡 GP - Within 24hrs'
        advice = 'See GP within 24hrs. Drink water, avoid spicy food.'
    else:
        urgency = '🟢 GP - Within 24hrs / Self-Care'
        advice = 'Rest, monitor symptoms. See GP if worse.'
    return dept, urgency, advice

with gr.Blocks(theme=gr.themes.Soft(), css=".footer {text-align:center; color:#666; font-size:13px; margin-top:20px;}") as demo:
    gr.Markdown(
        """
        # 🏥 Smart Symptom Triage System
        ### Helping Nurses Send Patients to the Right Department, Fast
        **Select up to 5 symptoms the patient is feeling, then press Triage Patient**
        """
    )
    with gr.Row():
        with gr.Column():
            s1 = gr.Dropdown(choices=symptom_list, label="Symptom 1 - Main symptom", value="Chest Pain")
            s2 = gr.Dropdown(choices=symptom_list, label="Symptom 2", value="Sweating")
            s3 = gr.Dropdown(choices=symptom_list, label="Symptom 3", value="Breathlessness")
            s4 = gr.Dropdown(choices=symptom_list, label="Symptom 4", value="No More Symptom")
            s5 = gr.Dropdown(choices=symptom_list, label="Symptom 5", value="No More Symptom")
            btn = gr.Button("🔍 TRIAGE PATIENT", variant="primary", size="lg")
            clear = gr.Button("Clear")
        with gr.Column():
            dept_out = gr.Textbox(label="🏥 Recommended Department", placeholder="Department will appear here...")
            urgency_out = gr.Textbox(label="⏰ Urgency Level", placeholder="Urgency will appear here...")
            advice_out = gr.Textbox(label="💡 Advice for Nurse / Patient", lines=3, placeholder="Advice will appear here...")
    btn.click(fn=triage_beautiful, inputs=[s1, s2, s3, s4, s5], outputs=[dept_out, urgency_out, advice_out])
    clear.click(fn=lambda: (None, None, None, None, None), outputs=[s1, s2, s3, s4, s5])
    gr.Markdown(
        """
        <div class="footer">
        <hr>
        <b>Smart Symptom Triage System | Final Year Dissertation Project</b><br>
        <b>Built by Francisca Kelechi</b><br>
        Developed for Hospital Triage Support | Model: Random Forest Classifier<br>
        ⚠️ <i>Disclaimer: This system is for triage support and education only. It does not replace professional medical judgement. In emergency, call emergency services immediately.</i><br>
        © 2026 Francisca Kelechi - All Rights Reserved
        </div>
        """
    )

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))