from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI()

templates = Jinja2Templates(directory="templates")


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("loan_model.pkl")


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict", response_class=HTMLResponse)
async def predict(

    request: Request,

    gender: str = Form(...),
    married: str = Form(...),
    dependents: str = Form(...),
    education: str = Form(...),
    self_employed: str = Form(...),

    applicant_income: float = Form(...),
    coapplicant_income: float = Form(...),

    loan_amount: float = Form(...),
    loan_amount_term: float = Form(...),

    credit_history: float = Form(...),

    property_area: str = Form(...)
):

    # --------------------------------------------------
    # Convert categorical values
    # --------------------------------------------------

    if gender == "Male":
        gender_value = 1
    else:
        gender_value = 0


    if married == "Yes":
        married_value = 1
    else:
        married_value = 0


    if education == "Graduate":
        education_value = 0
    else:
        education_value = 1


    if self_employed == "Yes":
        self_employed_value = 1
    else:
        self_employed_value = 0


    # Dependents

    if dependents == "3+":
        dependents_value = 3
    else:
        dependents_value = int(dependents)


    # Property Area

    property_area_mapping = {

        "Rural": 0,

        "Semiurban": 1,

        "Urban": 2
    }

    property_area_value = property_area_mapping[property_area]


    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    total_income = applicant_income + coapplicant_income

    total_applicant = total_income


    applicant_income_log = np.log(applicant_income + 1)

    loan_amount_log = np.log(loan_amount + 1)

    loan_amount_term_log = np.log(loan_amount_term + 1)

    total_income_log = np.log(total_income + 1)


    # --------------------------------------------------
    # Create model input
    # --------------------------------------------------

    input_data = pd.DataFrame({

        "Gender": [gender_value],

        "Married": [married_value],

        "Dependents": [dependents_value],

        "Education": [education_value],

        "Self_Employed": [self_employed_value],

        "Credit_History": [credit_history],

        "Property_Area": [property_area_value],

        "Total_Applicant": [total_applicant],

        "ApplicantIncomelog": [applicant_income_log],

        "LoanAmountlog": [loan_amount_log],

        "Loan_Amount_Termlog": [loan_amount_term_log],

        "AToatal_Income_log": [total_income_log]

    })


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(input_data)[0]


    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    if prediction == 1:

        result = "Loan Approved"

        result_class = "approved"

    else:

        result = "Loan Not Approved"

        result_class = "rejected"


    return templates.TemplateResponse(

        "index.html",

        {

            "request": request,

            "result": result,

            "result_class": result_class

        }

    )