import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image
import io
from lightgbm import LGBMClassifier
import category_encoders as ce
from imblearn.ensemble import EasyEnsembleClassifier
import shap
import plotly.express as px

# Load the pickled model and encoder
with open('best_model.pkl', 'rb') as model_file:
    model = pkl.load(model_file)

with open('cbe_encoder.pkl', 'rb') as encoder_file:
    encoder = pkl.load(encoder_file)

# Load the dataset for reference
data = pd.read_csv('brfss2022_data_wrangling_output.zip', compression='zip')
data['heart_disease'] = data['heart_disease'].apply(lambda x: 1 if x == 'yes' else 0).astype('int')

icon = Image.open("heart_disease.jpg")
st.set_page_config(layout='wide', page_title='AI-Powered Heart Disease Assessment', page_icon=icon)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style_v1.css")


def pick(label, mapping, default, help=None):
    keys = list(mapping.keys())
    labels = list(mapping.values())
    chosen = st.selectbox(label, labels, index=keys.index(default), help=help)
    return keys[labels.index(chosen)]


def yes_no(label, default="no", help=None):
    chosen = st.segmented_control(
        label,
        options=["Yes", "No"],
        default="Yes" if default == "yes" else "No",
        help=help,
    )
    return "yes" if (chosen or "No") == "Yes" else "no"


def section_head(number: str, kicker: str, title: str):
    st.markdown(
        f"""
        <div class="section-head">
          <div class="step-no">{number}</div>
          <div class="section-copy"><small>{kicker}</small><strong>{title}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


GENDER = {"female": "Female", "male": "Male", "nonbinary": "Non-binary"}
RACE = {
    "white_only_non_hispanic": "White, non-Hispanic",
    "black_only_non_hispanic": "Black, non-Hispanic",
    "asian_only_non_hispanic": "Asian, non-Hispanic",
    "american_indian_or_alaskan_native_only_non_hispanic": "American Indian or Alaska Native, non-Hispanic",
    "multiracial_non_hispanic": "Multiracial, non-Hispanic",
    "hispanic": "Hispanic",
    "native_hawaiian_or_other_pacific_islander_only_non_hispanic": "Native Hawaiian or other Pacific Islander, non-Hispanic",
}
AGE = {
    "Age_18_to_24": "18–24",
    "Age_25_to_29": "25–29",
    "Age_30_to_34": "30–34",
    "Age_35_to_39": "35–39",
    "Age_40_to_44": "40–44",
    "Age_45_to_49": "45–49",
    "Age_50_to_54": "50–54",
    "Age_55_to_59": "55–59",
    "Age_60_to_64": "60–64",
    "Age_65_to_69": "65–69",
    "Age_70_to_74": "70–74",
    "Age_75_to_79": "75–79",
    "Age_80_or_older": "80 or older",
}
HEALTH = {"excellent": "Excellent", "very_good": "Very good", "good": "Good", "fair": "Fair", "poor": "Poor"}
PROVIDER = {"yes_only_one": "Yes, one provider", "more_than_one": "More than one", "no": "No"}
ASTHMA = {"never_asthma": "Never", "current_asthma": "Current", "former_asthma": "Former"}
DIABETES = {"yes": "Yes", "no": "No", "no_prediabetes": "Prediabetes / borderline", "yes_during_pregnancy": "Yes, during pregnancy"}
BMI = {
    "underweight_bmi_less_than_18_5": "Underweight (BMI < 18.5)",
    "normal_weight_bmi_18_5_to_24_9": "Normal (18.5–24.9)",
    "overweight_bmi_25_to_29_9": "Overweight (25–29.9)",
    "obese_bmi_30_or_more": "Obesity (BMI ≥ 30)",
}
CHECKUP = {
    "past_year": "Within the past year",
    "past_2_years": "Within the past 2 years",
    "past_5_years": "Within the past 5 years",
    "5+_years_ago": "5 or more years ago",
    "never": "Never",
}
DAYS_NOT_GOOD = {
    "zero_days_not_good": "None",
    "1_to_13_days_not_good": "1–13 days",
    "14_plus_days_not_good": "14 or more days",
}
SMOKING = {
    "never_smoked": "Never smoked",
    "former_smoker": "Former smoker",
    "current_smoker_some_days": "Current smoker, some days",
    "current_smoker_every_day": "Current smoker, every day",
}
SLEEP = {
    "very_short_sleep_0_to_3_hours": "0–3 hours",
    "short_sleep_4_to_5_hours": "4–5 hours",
    "normal_sleep_6_to_8_hours": "6–8 hours",
    "long_sleep_9_to_10_hours": "9–10 hours",
    "very_long_sleep_11_or_more_hours": "11 or more hours",
}
DRINKS = {
    "did_not_drink": "Did not drink",
    "very_low_consumption_0.01_to_1_drinks": "Up to 1 drink / week",
    "low_consumption_1.01_to_5_drinks": "1–5 drinks / week",
    "moderate_consumption_5.01_to_10_drinks": "5–10 drinks / week",
    "high_consumption_10.01_to_20_drinks": "10–20 drinks / week",
    "very_high_consumption_more_than_20_drinks": "More than 20 drinks / week",
}

st.markdown(
    """
    <div class="hero-banner">
      <div class="hero-top"><span class="pulse"></span> Cardio risk studio</div>
      <h1>Know your heart risk in a few considered answers.</h1>
      <p>A clinical-style intake with plain-language choices. Behind the scenes, the original coded features still drive the model.</p>
      <div class="hero-steps">
        <div class="hero-step"><b>01  Profile</b><span>Age, sex and ethnicity</span></div>
        <div class="hero-step"><b>02  History</b><span>Conditions and check-ups</span></div>
        <div class="hero-step"><b>03  Living</b><span>Sleep, movement, smoking, alcohol</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section_head("1", "Demographics", "Who is being assessed")
with st.container(border=True):
    d1, d2, d3 = st.columns(3)
    with d1:
        gender = pick("Gender", GENDER, "male")
    with d2:
        race = pick("Race / ethnicity", RACE, "white_only_non_hispanic")
    with d3:
        age_category = pick("Age group", AGE, "Age_40_to_44")

section_head("2", "Clinical record", "Medical history")
with st.container(border=True):
    m1, m2 = st.columns(2)
    with m1:
        general_health = pick("Overall health", HEALTH, "excellent")
        heart_attack = yes_no("Ever diagnosed with a heart attack?", help="A heart attack occurs when blood flow to part of the heart is blocked.")
        kidney_disease = yes_no("Doctor ever told you that you have kidney disease?")
        asthma = pick("Asthma", ASTHMA, "never_asthma")
        could_not_afford_to_see_doctor = yes_no("Unable to see a doctor due to cost?")
        health_care_provider = pick("Primary health-care provider", PROVIDER, "yes_only_one")
        walking = yes_no("Difficulty walking or climbing stairs?")
    with m2:
        stroke = yes_no("Ever diagnosed with a stroke?", help="A stroke happens when blood supply to part of the brain is interrupted.")
        diabetes = pick("Diabetes", DIABETES, "no")
        bmi = pick("BMI category", BMI, "normal_weight_bmi_18_5_to_24_9", help="BMI is weight relative to height. Calculator: https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm")
        length_of_time_since_last_routine_checkup = pick("Last routine checkup", CHECKUP, "past_year")
        depressive_disorder = yes_no("Doctor ever told you that you have a depressive disorder?", help="Persistent sadness, loss of interest, or related symptoms diagnosed by a clinician.")
        physical_health = pick("Days physical health was not good (past 30 days)", DAYS_NOT_GOOD, "zero_days_not_good")
        mental_health = pick("Days mental health was not good (past 30 days)", DAYS_NOT_GOOD, "zero_days_not_good")

section_head("3", "Daily life", "Lifestyle")
with st.container(border=True):
    l1, l2 = st.columns(2)
    with l1:
        smoking_status = pick("Smoking status", SMOKING, "never_smoked")
        sleep_category = pick("Typical nightly sleep", SLEEP, "normal_sleep_6_to_8_hours")
        exercise_status = yes_no("Exercised in the past 30 days?", default="yes")
    with l2:
        drinks_category = pick("Alcoholic drinks in a typical week", DRINKS, "did_not_drink")
        binge_drinking_status = yes_no(
            "Binge drinking in the past 30 days?",
            help="5+ drinks for men or 4+ for women in about 2 hours.",
        )

input_data = {
    'gender': gender,
    'race': race,
    'general_health': general_health,
    'health_care_provider': health_care_provider,
    'could_not_afford_to_see_doctor': could_not_afford_to_see_doctor,
    'length_of_time_since_last_routine_checkup': length_of_time_since_last_routine_checkup,
    'ever_diagnosed_with_heart_attack': heart_attack,
    'ever_diagnosed_with_a_stroke': stroke,
    'ever_told_you_had_a_depressive_disorder': depressive_disorder,
    'ever_told_you_have_kidney_disease': kidney_disease,
    'ever_told_you_had_diabetes': diabetes,
    'BMI': bmi,
    'difficulty_walking_or_climbing_stairs': walking,
    'physical_health_status': physical_health,
    'mental_health_status': mental_health,
    'asthma_Status': asthma,
    'smoking_status': smoking_status,
    'binge_drinking_status': binge_drinking_status,
    'exercise_status_in_past_30_Days': exercise_status,
    'age_category': age_category,
    'sleep_category': sleep_category,
    'drinks_category': drinks_category
}

def class1_shap_matrix(shap_values):
    values = shap_values
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def predict_heart_disease_risk(input_data, model, encoder):
    input_df = pd.DataFrame([input_data])
    input_encoded = encoder.transform(input_df, y=None, override_return_df=False)
    prediction = model.predict_proba(input_encoded)[:, 1][0] * 100
    return prediction

section_head("4", "Results", "Run the assessment")
result_left, result_right = st.columns((1.15, 1))

with result_left:
    btn1 = st.button("Get heart-disease risk assessment", type="primary", use_container_width=True)

if btn1:
    try:
        risk = predict_heart_disease_risk(input_data, model, encoder)
        with result_left:
            band = "Very high" if risk > 70 else "High" if risk > 40 else "Moderate" if risk > 25 else "Low"
            st.metric("Predicted heart-disease risk", f"{risk:.1f}%")
            st.markdown(f"**Risk band:** {band}")
            input_df = pd.DataFrame([input_data])
            input_encoded = encoder.transform(input_df, y=None, override_return_df=False)
            lgbm_model = model.estimators_[0].steps[-1][1]
            explainer = shap.TreeExplainer(lgbm_model)
            shap_matrix = class1_shap_matrix(explainer.shap_values(input_encoded))
            feature_importances = np.abs(shap_matrix).sum(axis=0)
            total = feature_importances.sum()
            if total == 0:
                feature_importances = np.ones_like(feature_importances) / len(feature_importances) * 100
            else:
                feature_importances = feature_importances / total * 100
            feature_importance_df = pd.DataFrame({
                'Feature': input_encoded.columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            recommendations = []
            if risk > 70:
                recommendations.append("Your risk of heart disease is very high. Here are some recommendations to reduce your risk:")
            elif risk > 40:
                recommendations.append("Your risk of heart disease is high. Here are some recommendations to reduce your risk:")
            elif risk > 25:
                recommendations.append("Your risk of heart disease is moderate. Here are some recommendations to reduce your risk:")
            else:
                recommendations.append("Your risk of heart disease is low. Keep up the good work and continue to maintain a healthy lifestyle.")

            if risk > 25:
                cumulative_importance = 0
                important_features = set()
                for index, row in feature_importance_df.iterrows():
                    cumulative_importance += row['Importance']
                    important_features.add(row['Feature'])
                    if cumulative_importance >= 50:
                        break
                
                # Ensure unique features are added only once
                additional_features = [
                    ('ever_told_you_had_diabetes', diabetes == "yes"),
                    ('ever_diagnosed_with_heart_attack', heart_attack == "yes"),
                    ('ever_told_you_had_a_depressive_disorder', depressive_disorder == "yes"),
                    ('ever_diagnosed_with_a_stroke', stroke == "yes"),
                    ('age_category', age_category in ["Age_55_to_59", "Age_60_to_64", "Age_65_to_69", "Age_70_to_74", "Age_75_to_79", "Age_80_or_older"]),
                    ('length_of_time_since_last_routine_checkup', length_of_time_since_last_routine_checkup in ["Age_55_to_59", "Age_60_to_64", "Age_65_to_69", "Age_70_to_74", "Age_75_to_79", "Age_80_or_older"]),
                    ('general_health', general_health in ["fair", "poor"]),
                    ('BMI', bmi in ["overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"]),
                    ('smoking_status', smoking_status != "never_smoked"),
                    ('exercise_status_in_past_30_Days', exercise_status == "no"),
                    ('binge_drinking_status', binge_drinking_status == "yes"),
                    ('drinks_category', drinks_category in ["high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"]),
                    ('sleep_category', sleep_category in ["short_sleep_4_to_5_hours", "very_short_sleep_0_to_3_hours"]),
                    ('physical_health_status', physical_health in ["1_to_13_days_not_good", "14_plus_days_not_good"]),
                    ('mental_health_status', mental_health in ["1_to_13_days_not_good", "14_plus_days_not_good"]),
                    ('asthma_Status', asthma in ["current_asthma", "former_asthma"]),
                    ('difficulty_walking_or_climbing_stairs', walking == "yes"),
                    ('length_of_time_since_last_routine_checkup', length_of_time_since_last_routine_checkup != "past_year"),
                    ('could_not_afford_to_see_doctor', could_not_afford_to_see_doctor == "yes"),
                    ('health_care_provider', health_care_provider == "no"),
                    ('ever_told_you_have_kidney_disease', kidney_disease == "yes")
                ]

                for feature, condition in additional_features:
                    if condition:
                        important_features.add(feature)

                # Mapping for feature names to user-friendly names
                feature_name_mapping = {
                    'ever_diagnosed_with_heart_attack': 'Heart Attack',
                    'general_health': 'General Health',
                    'ever_diagnosed_with_a_stroke': 'Stroke',
                    'ever_told_you_have_kidney_disease': 'Kidney Disease',
                    'ever_told_you_had_diabetes': 'Diabetes',
                    'physical_health_status': 'Physical Health',
                    'ever_told_you_had_a_depressive_disorder': 'Depression',
                    'sleep_category': 'Sleep',
                    'age_category': 'Age',
                    'length_of_time_since_last_routine_checkup': 'Checkup Time',
                    'BMI': 'BMI',
                    'smoking_status': 'Smoking',
                    'exercise_status_in_past_30_Days': 'Exercise',
                    'binge_drinking_status': 'Binge Drinking',
                    'drinks_category': 'Alcohol',
                    'could_not_afford_to_see_doctor': 'Doctor Access',
                    'health_care_provider': 'Healthcare Provider',
                    'asthma_Status': 'Asthma',
                    'difficulty_walking_or_climbing_stairs': 'Mobility',
                    'mental_health_status': 'Mental Health',
                }

                # Ensure that the features with recommendations are included in the final features list
                final_features = []
                feature_to_recommendation = {}
                for feature in important_features:
                    importance = feature_importance_df.loc[feature_importance_df['Feature'] == feature, 'Importance'].values[0]
                    if feature == 'ever_diagnosed_with_heart_attack' and heart_attack == "yes":
                        recommendation = f"- History of heart attack contributed {importance:.2f}% to your risk. Regularly visit your cardiologist and adhere to prescribed medications. Monitor any new or worsening symptoms and seek immediate medical attention if needed."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'ever_diagnosed_with_a_stroke' and stroke == "yes":
                        recommendation = f"- History of stroke contributed {importance:.2f}% to your risk. Follow your neurologist's recommendations and take prescribed medications consistently. Engage in approved physical therapy or exercises to regain strength and mobility."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'age_category' and age_category in ["Age_55_to_59", "Age_60_to_64", "Age_65_to_69", "Age_70_to_74", "Age_75_to_79", "Age_80_or_older"]:
                        recommendation = f"- Age category contributed {importance:.2f}% to your risk. While you can't change your age, maintaining a healthy lifestyle can mitigate risks associated with aging. Ensure regular check-ups, eat a balanced diet, stay active, and avoid smoking."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'general_health' and general_health in ["fair", "poor"]:
                        recommendation = f"- General health contributed {importance:.2f}% to your risk. Focus on improving your overall health through a balanced diet and regular check-ups."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'ever_told_you_have_kidney_disease' and kidney_disease == "yes":
                        recommendation = f"- Kidney disease contributed {importance:.2f}% to your risk. Regularly monitor your kidney function and follow your doctor's advice to manage your condition. Stay hydrated and maintain a kidney-friendly diet."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'ever_told_you_had_diabetes' and diabetes == "yes":
                        recommendation = f"- Diabetes contributed {importance:.2f}% to your risk. Manage your diabetes through diet, exercise, and medication as prescribed by your doctor."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'smoking_status' and smoking_status != "never_smoked":
                        recommendation = f"- Smoking status contributed {importance:.2f}% to your risk. Quit smoking to significantly reduce your risk of heart disease."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'exercise_status_in_past_30_Days' and exercise_status == "no":
                        recommendation = f"- Lack of exercise contributed {importance:.2f}% to your risk. Engage in regular physical activity to improve your heart health."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'binge_drinking_status' and binge_drinking_status == "yes":
                        recommendation = f"- Binge drinking contributed {importance:.2f}% to your risk. Reducing or eliminating alcohol consumption can significantly lower your risk of heart disease. Consider seeking support for alcohol moderation or cessation if needed."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'drinks_category' and drinks_category in ["high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"]:
                        recommendation = f"- Alcohol consumption contributed {importance:.2f}% to your risk. Limit alcohol consumption to lower your risk."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'sleep_category' and sleep_category in ["short_sleep_4_to_5_hours", "very_short_sleep_0_to_3_hours"]:
                        recommendation = f"- Sleep category contributed {importance:.2f}% to your risk. Consider aiming for 7-9 hours of quality sleep each night. Adequate sleep is crucial for maintaining heart health."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'physical_health_status' and physical_health in ["1_to_13_days_not_good", "14_plus_days_not_good"]:
                        recommendation = f"- Physical health contributed {importance:.2f}% to your risk. Engage in regular physical activity and consult a healthcare provider if you have persistent physical health issues."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'mental_health_status' and mental_health in ["1_to_13_days_not_good", "14_plus_days_not_good"]:
                        recommendation = f"- Mental health contributed {importance:.2f}% to your risk. Consider seeking support from a mental health professional and practice stress-reducing activities."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'asthma_Status' and asthma in ["current_asthma", "former_asthma"]:
                        recommendation = f"- Asthma contributed {importance:.2f}% to your risk. Manage your asthma by following your treatment plan, avoiding asthma triggers, and using your medications as prescribed."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'ever_told_you_had_a_depressive_disorder' and depressive_disorder == "yes":
                        recommendation = f"- Depressive disorder contributed {importance:.2f}% to your risk. Consider seeking support from a mental health professional, practicing stress-reducing activities, and maintaining a healthy lifestyle to manage depressive symptoms."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'difficulty_walking_or_climbing_stairs' and walking == "yes":
                        recommendation = f"- Difficulty walking or climbing stairs contributed {importance:.2f}% to your risk. Consider consulting with a healthcare provider for appropriate interventions and exercises to improve mobility and strength."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'length_of_time_since_last_routine_checkup' and length_of_time_since_last_routine_checkup != "past_year":
                        recommendation = f"- Time since last routine checkup contributed {importance:.2f}% to your risk. Regular health checkups are important for early detection and management of health conditions. Schedule regular appointments with your healthcare provider to monitor and maintain your heart health."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'could_not_afford_to_see_doctor' and could_not_afford_to_see_doctor == "yes":
                        recommendation = f"- Difficulty affording to see a doctor contributed {importance:.2f}% to your risk. Explore community health services, sliding scale clinics, or health insurance options to ensure you have access to necessary medical care."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'health_care_provider' and health_care_provider == "no":
                        recommendation = f"- Not having a primary health care provider contributed {importance:.2f}% to your risk. Establishing a relationship with a primary care provider can help manage and prevent health issues. Consider finding a primary health care provider to ensure regular check-ups and consistent medical advice."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    if feature == 'BMI' and bmi in ["overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"]:
                        recommendation = f"- BMI contributed {importance:.2f}% to your risk. Maintaining a healthy weight through a balanced diet and regular exercise can help reduce your risk of heart disease. Consider consulting a healthcare provider for personalized advice."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)    

                # Calculate the remaining contribution for "Other Factors"
                total_importance = sum([feature_importance_df.loc[feature_importance_df['Feature'] == feature, 'Importance'].values[0] for feature in final_features])
                other_factors_importance = 100 - total_importance

                # Prepare data for the pie chart
                pie_data = {
                    'Feature': [feature_name_mapping[feature] for feature in final_features] + ['Other Factors'],
                    'Importance': [feature_importance_df.loc[feature_importance_df['Feature'] == feature, 'Importance'].values[0] for feature in final_features] + [other_factors_importance]
                }
                pie_df = pd.DataFrame(pie_data)

                # Create the pie chart
                fig = px.pie(pie_df, names='Feature', values='Importance', color_discrete_sequence=["#9f1239", "#fb7185", "#0e7490", "#334155", "#fda4af", "#155e75", "#64748b"])

                # Display the pie chart
                with result_right:
                    st.subheader("Contribution to risk")
                    st.plotly_chart(fig, use_container_width=True)

                # Display recommendations in sorted order
                sorted_recommendations = sorted([(feature, feature_to_recommendation[feature]) for feature in final_features], key=lambda x: feature_importance_df.loc[feature_importance_df['Feature'] == x[0], 'Importance'].values[0], reverse=True)
                for feature, recommendation in sorted_recommendations:
                    st.write(recommendation)
            else:
                st.write("Your risk of heart disease is low. Keep up the good work and continue to maintain a healthy lifestyle.")

    except Exception as e:
        result_left.error(e)
