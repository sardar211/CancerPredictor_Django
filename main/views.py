from django.shortcuts import render, redirect
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

# Create your views here.


def home(request):
    return render(request, "home.html")


def predic_lung_cancer(
    age,
    gender,
    air_pollution,
    dust_allergy,
    occu_pational_hazards,
    genetic_risk,
    chronic_lung_disease,
    balanced_diet,
    obesity,
    smoking,
    alcohol_consuming,
    passive_smoker,
    coughing_of_blood,
    fatigue,
    weight_loss,
    shortness_of_breath,
    wheezing,
    swallowing_difficulty,
    clubbing_of_finger_nails,
    frequent_cold,
    dry_cough,
    snoring,
    chest_pain,
):

    new_data = pd.DataFrame(
        {
            "age": [age],
            "gender": [gender],
            "air_pollution": [air_pollution],
            "alcohol_use": [alcohol_consuming],
            "dust_allergy": [dust_allergy],
            "occupational_hazards": [occu_pational_hazards],
            "genetic_risk": [genetic_risk],
            "chronic_lung_disease": [chronic_lung_disease],
            "balanced_diet": [balanced_diet],
            "obesity": [obesity],
            "smoking": [smoking],
            "passive_smoker": [passive_smoker],
            "chest_pain": [chest_pain],
            "coughing_of_blood": [coughing_of_blood],
            "fatigue": [fatigue],
            "weight_loss": [weight_loss],
            "shortness_of_breath": [shortness_of_breath],
            "wheezing": [wheezing],
            "swallowing_difficulty": [swallowing_difficulty],
            "clubbing_of_finger_nails": [clubbing_of_finger_nails],
            "frequent_cold": [frequent_cold],
            "dry_cough": [dry_cough],
            "snoring": [snoring],
        }
    )

    decision_tree_model = joblib.load("decision_tree_model.joblib")
    scaler = joblib.load("scaler.joblib")

    scaled_new_data = scaler.transform(new_data)

    warnings.filterwarnings(
        "ignore", category=UserWarning, module="sklearn.base"
    )

    predicted_lung_cancer_level = decision_tree_model.predict(scaled_new_data)[
        0
    ]

    return predicted_lung_cancer_level


def form(request):
    if request.method == "POST":
        print("Here")

        age = request.POST.get("age")
        gender = request.POST.get("gender")
        air_pollution = request.POST.get("air_pollution")
        dust_allergy = request.POST.get("dust_allergy")
        occu_pational_hazards = request.POST.get("occu_pational_hazards")
        genetic_risk = request.POST.get("genetic_risk")
        chronic_lung_disease = request.POST.get("chronic_lung_disease")
        balanced_diet = request.POST.get("balanced_diet")
        obesity = request.POST.get("obesity")
        smoking = request.POST.get("smoking")
        alcohol_consuming = request.POST.get("alcohol_consuming")
        passive_smoker = request.POST.get("passive_smoker")
        coughing_of_blood = request.POST.get("coughing_of_blood")
        fatigue = request.POST.get("fatigue")
        weight_loss = request.POST.get("weight_loss")
        shortness_of_breath = request.POST.get("shortness_of_breath")
        wheezing = request.POST.get("wheezing")
        swallowing_difficulty = request.POST.get("swallowing_difficulty")
        clubbing_of_finger_nails = request.POST.get("clubbing_of_finger_nails")
        frequent_cold = request.POST.get("frequent_cold")
        dry_cough = request.POST.get("dry_cough")
        snoring = request.POST.get("snoring")
        chest_pain = request.POST.get("chest_pain")

        level = predic_lung_cancer(
            age=age,
            gender=gender,
            air_pollution=air_pollution,
            dust_allergy=dust_allergy,
            occu_pational_hazards=occu_pational_hazards,
            genetic_risk=genetic_risk,
            chronic_lung_disease=chronic_lung_disease,
            balanced_diet=balanced_diet,
            obesity=obesity,
            smoking=smoking,
            alcohol_consuming=alcohol_consuming,
            passive_smoker=passive_smoker,
            coughing_of_blood=coughing_of_blood,
            fatigue=fatigue,
            weight_loss=weight_loss,
            shortness_of_breath=shortness_of_breath,
            wheezing=wheezing,
            swallowing_difficulty=swallowing_difficulty,
            clubbing_of_finger_nails=clubbing_of_finger_nails,
            frequent_cold=frequent_cold,
            dry_cough=dry_cough,
            snoring=snoring,
            chest_pain=chest_pain,
        )

        # print("Level = ", level)

        return redirect("prediction", level=level)

    return render(request, "form.html")


def prediction(request, level):
    cancer_level = ""
    cancer_message = ""

    if level == 0:
        cancer_level = "Low"
        cancer_message = "Your risk of having lung cancer is low. However, it is always a good idea to maintain regular check-ups with your healthcare provider."
    elif level == 1:
        cancer_level = "Medium"
        cancer_message = "You have a moderate risk of lung cancer. It is important to consult your doctor for a detailed examination and to discuss possible preventive measures."
    elif level == 2:
        cancer_level = "High"
        cancer_message = "You have a high risk of lung cancer. Please seek immediate medical advice and undergo further diagnostic tests as recommended by your healthcare provider."
    else:
        cancer_level = "Invalid Input"
        cancer_message = "There was an issue with the data provided. Please ensure all inputs are correct and try again."

    return render(
        request,
        "prediction.html",
        {"cancer_level": cancer_level, "cancer_message": cancer_message},
    )
