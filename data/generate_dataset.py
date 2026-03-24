"""
Generate a synthetic Telco Customer Churn dataset.
Produces ~7000 rows with realistic correlations between features and churn.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

N = 7043  # Match real Telco dataset size

# --- Customer IDs ---
customer_ids = [f"{i:04d}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for i in range(1, N + 1)]

# --- Demographics ---
gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])

# --- Tenure (months) ---
# Bimodal: many new + many long-term customers
tenure = np.concatenate([
    np.random.exponential(scale=8, size=int(N * 0.4)).astype(int).clip(1, 72),
    np.random.normal(loc=50, scale=15, size=int(N * 0.35)).astype(int).clip(1, 72),
    np.random.randint(1, 73, size=N - int(N * 0.4) - int(N * 0.35))
])
np.random.shuffle(tenure)

# --- Services ---
phone_service = np.random.choice(["Yes", "No"], N, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    np.random.choice(["Yes", "No"], N, p=[0.42, 0.58])
)

internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])

def internet_dependent_service(internet_service, n):
    """For services that depend on having internet."""
    return np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n, p=[0.40, 0.60])
    )

online_security = internet_dependent_service(internet_service, N)
online_backup = internet_dependent_service(internet_service, N)
device_protection = internet_dependent_service(internet_service, N)
tech_support = internet_dependent_service(internet_service, N)
streaming_tv = internet_dependent_service(internet_service, N)
streaming_movies = internet_dependent_service(internet_service, N)

# --- Contract ---
contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24]
)

# --- Billing ---
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])

payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

# --- Monthly Charges ---
base_charge = np.random.uniform(18, 30, N)
fiber_bump = np.where(internet_service == "Fiber optic", np.random.uniform(25, 40, N), 0)
dsl_bump = np.where(internet_service == "DSL", np.random.uniform(10, 20, N), 0)
service_count = sum([
    (online_security == "Yes").astype(float),
    (online_backup == "Yes").astype(float),
    (device_protection == "Yes").astype(float),
    (tech_support == "Yes").astype(float),
    (streaming_tv == "Yes").astype(float),
    (streaming_movies == "Yes").astype(float),
])
service_bump = service_count * np.random.uniform(5, 10, N)
monthly_charges = np.round(base_charge + fiber_bump + dsl_bump + service_bump + np.random.normal(0, 3, N), 2)
monthly_charges = monthly_charges.clip(18.25, 118.75)

# --- Total Charges ---
total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, N), 2)
total_charges = total_charges.clip(18.8, 8700)

# Introduce a few blanks in TotalCharges (like the real dataset)
blank_indices = np.random.choice(N, size=11, replace=False)

# --- Churn (target) ---
# Build a churn probability based on realistic risk factors
churn_prob = np.full(N, 0.15)  # base rate

# Higher churn for short tenure
churn_prob += np.where(tenure <= 6, 0.25, 0)
churn_prob += np.where((tenure > 6) & (tenure <= 12), 0.10, 0)

# Month-to-month contracts churn more
churn_prob += np.where(contract == "Month-to-month", 0.15, 0)
churn_prob -= np.where(contract == "Two year", 0.10, 0)

# Fiber optic customers churn more (higher bills, more competition)
churn_prob += np.where(internet_service == "Fiber optic", 0.08, 0)

# Electronic check payment correlates with churn
churn_prob += np.where(payment_method == "Electronic check", 0.05, 0)

# No online security / tech support increases churn
churn_prob += np.where(online_security == "No", 0.03, 0)
churn_prob += np.where(tech_support == "No", 0.03, 0)

# Senior citizens churn slightly more
churn_prob += np.where(senior_citizen == 1, 0.05, 0)

# High monthly charges increase churn
churn_prob += np.where(monthly_charges > 80, 0.05, 0)

# Paperless billing slight bump
churn_prob += np.where(paperless_billing == "Yes", 0.02, 0)

# Clip to valid range
churn_prob = churn_prob.clip(0.02, 0.95)

# Generate churn decisions
churn = np.array(["Yes" if np.random.random() < p else "No" for p in churn_prob])

# --- Build DataFrame ---
df = pd.DataFrame({
    "customerID": customer_ids,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges.astype(str),
    "Churn": churn,
})

# Insert blank TotalCharges for realism (real dataset has 11 blanks)
for idx in blank_indices:
    df.at[idx, "TotalCharges"] = " "

# --- Save ---
output_path = os.path.join(os.path.dirname(__file__), "telco_churn.csv")
df.to_csv(output_path, index=False)

print(f"Dataset generated: {output_path}")
print(f"Shape: {df.shape}")
print(f"Churn rate: {(churn == 'Yes').mean():.2%}")
print(f"\nSample:\n{df.head()}")
