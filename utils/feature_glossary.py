"""
Plain-English glossary for diabetes and heart disease model features.
"""

from __future__ import annotations

DIABETES_GLOSSARY = {
    "Pregnancies": {
        "label": "Number of Pregnancies",
        "plain": "How many times you have been pregnant. Higher numbers can be associated with gestational diabetes history.",
        "normal_range": "0–5 is typical",
        "unit": "count",
        "why_it_matters": "Multiple pregnancies can affect insulin resistance over time.",
    },
    "Glucose": {
        "label": "Fasting Blood Glucose",
        "plain": "Your blood sugar level after not eating for 8+ hours. This is the single strongest predictor of diabetes risk.",
        "normal_range": "Below 100 mg/dL is normal. 100–125 is pre-diabetic. 126+ may indicate diabetes.",
        "unit": "mg/dL",
        "why_it_matters": "High glucose means your body isn't processing sugar efficiently.",
    },
    "BloodPressure": {
        "label": "Resting Blood Pressure",
        "plain": "The pressure in your arteries when your heart is at rest (diastolic reading).",
        "normal_range": "Below 80 mm Hg is normal. 80–89 is elevated. 90+ is high.",
        "unit": "mm Hg",
        "why_it_matters": "Consistently high blood pressure strains your cardiovascular system and is linked to metabolic disorders.",
    },
    "SkinThickness": {
        "label": "Skin Fold Thickness",
        "plain": "A measure of body fat taken from a skin fold at the tricep.",
        "normal_range": "Typically 10–40mm for most adults",
        "unit": "mm",
        "why_it_matters": "Higher skin thickness can indicate excess body fat, a risk factor for insulin resistance.",
    },
    "Insulin": {
        "label": "2-Hour Serum Insulin",
        "plain": "Insulin level in your blood 2 hours after consuming glucose.",
        "normal_range": "16–166 μU/mL is considered normal",
        "unit": "μU/mL",
        "why_it_matters": "Abnormal insulin levels indicate how well your body responds to blood sugar.",
    },
    "BMI": {
        "label": "Body Mass Index",
        "plain": "A measure of body fat based on your height and weight.",
        "normal_range": "18.5–24.9 is healthy. 25–29.9 is overweight. 30+ is obese.",
        "unit": "kg/m²",
        "why_it_matters": "Higher BMI significantly increases the risk of type 2 diabetes.",
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Family History Score",
        "plain": "A score that reflects how much your family history of diabetes influences your genetic risk.",
        "normal_range": "0.0–2.5. Higher values mean stronger family history.",
        "unit": "score",
        "why_it_matters": "Genetics plays a meaningful role in diabetes susceptibility.",
    },
    "Age": {
        "label": "Age",
        "plain": "Your age in years. Risk of type 2 diabetes increases with age.",
        "normal_range": "Risk increases notably after age 45.",
        "unit": "years",
        "why_it_matters": "Insulin sensitivity naturally decreases as we age.",
    },
}

HEART_GLOSSARY = {
    "age": {
        "label": "Age",
        "plain": "Your age in years, which is a major baseline risk factor in cardiovascular disease.",
        "normal_range": "Risk rises with age, especially after 45–50 years.",
        "unit": "years",
        "why_it_matters": "Vascular stiffness and cumulative risk exposures increase over time.",
    },
    "sex": {
        "label": "Sex",
        "plain": "Biological sex coded as 0 (female) or 1 (male) in this dataset.",
        "normal_range": "Categorical feature (0/1)",
        "unit": "category",
        "why_it_matters": "Heart disease patterns and prevalence differ by sex across populations.",
    },
    "cp": {
        "label": "Chest Pain Type",
        "plain": "Type of chest pain pattern (0–3), from typical angina to asymptomatic presentations.",
        "normal_range": "Categorical feature (0–3)",
        "unit": "category",
        "why_it_matters": "Specific chest pain patterns can signal underlying coronary disease likelihood.",
    },
    "trestbps": {
        "label": "Resting Blood Pressure",
        "plain": "Blood pressure measured while at rest before exercise testing.",
        "normal_range": "Around 90–120 mm Hg is typical for systolic values.",
        "unit": "mm Hg",
        "why_it_matters": "Elevated blood pressure increases strain on arteries and the heart.",
    },
    "chol": {
        "label": "Serum Cholesterol",
        "plain": "Total cholesterol concentration in blood.",
        "normal_range": "Below 200 mg/dL is desirable; 200+ is elevated.",
        "unit": "mg/dL",
        "why_it_matters": "Higher cholesterol contributes to plaque formation and cardiovascular risk.",
    },
    "fbs": {
        "label": "Fasting Blood Sugar >120",
        "plain": "Binary marker showing whether fasting blood sugar is above 120 mg/dL.",
        "normal_range": "Categorical feature (0/1)",
        "unit": "category",
        "why_it_matters": "Elevated fasting glucose is linked with metabolic dysfunction and vascular risk.",
    },
    "restecg": {
        "label": "Resting ECG Result",
        "plain": "Resting electrocardiogram category indicating baseline electrical heart findings.",
        "normal_range": "Categorical feature (0–2)",
        "unit": "category",
        "why_it_matters": "Abnormal ECG patterns can reflect structural or ischemic heart changes.",
    },
    "thalach": {
        "label": "Maximum Heart Rate",
        "plain": "Highest heart rate achieved during stress testing.",
        "normal_range": "Varies by age; generally higher achievable rates suggest better exercise tolerance.",
        "unit": "beats/min",
        "why_it_matters": "Reduced exercise capacity can indicate impaired cardiovascular function.",
    },
    "exang": {
        "label": "Exercise-Induced Angina",
        "plain": "Whether chest pain is triggered by physical exertion (0/1).",
        "normal_range": "Categorical feature (0/1)",
        "unit": "category",
        "why_it_matters": "Exertional angina is a classic signal of possible coronary perfusion limits.",
    },
    "oldpeak": {
        "label": "ST Depression (Oldpeak)",
        "plain": "Magnitude of ST depression during exercise relative to rest on ECG.",
        "normal_range": "Near 0 is typical; higher values indicate greater abnormality.",
        "unit": "score",
        "why_it_matters": "Higher ST depression is associated with ischemia and worse cardiac risk profiles.",
    },
    "slope": {
        "label": "Slope of Peak ST Segment",
        "plain": "Categorical representation of the ST segment slope during peak exercise.",
        "normal_range": "Categorical feature (0–2)",
        "unit": "category",
        "why_it_matters": "ST slope pattern contributes to ischemia interpretation and risk stratification.",
    },
    "ca": {
        "label": "Major Vessels Colored",
        "plain": "Number of major vessels visualized by fluoroscopy.",
        "normal_range": "0–4 in this dataset",
        "unit": "count",
        "why_it_matters": "More affected/visible vessels can reflect more extensive coronary involvement.",
    },
    "thal": {
        "label": "Thalassemia Test Result",
        "plain": "Categorical thalassemia-related perfusion status used in classic cardiac datasets.",
        "normal_range": "Categorical feature (0–3)",
        "unit": "category",
        "why_it_matters": "Abnormal thal patterns are associated with perfusion defects and higher risk.",
    },
}

GLOSSARY = {**DIABETES_GLOSSARY, **HEART_GLOSSARY}
