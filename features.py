import pandas as pd
import numpy as np

def engineer_features(age, monthly_income, num_dependents,
                       revolving_utilization, debt_ratio,
                       num_open_credit_lines, num_real_estate_loans,
                       num_30_59_days_late, num_60_89_days_late,
                       num_90_days_late):
    """
    Takes raw applicant inputs and returns a single-row DataFrame
    matching the 27 features the XGBoost model was trained on.
    """
    import pandas as pd
    import numpy as np

    # Handle missing income (mirrors Phase 4's imputation logic)
    monthly_income_missing = 1 if monthly_income is None or monthly_income == 0 else 0
    if monthly_income_missing:
        monthly_income_value = 5400  # median income used in training imputation
    else:
        monthly_income_value = monthly_income

    # Delinquency aggregates (Phase 5)
    total_past_due = num_30_59_days_late + num_60_89_days_late + num_90_days_late
    ever_delinquent = 1 if total_past_due > 0 else 0
    worst_delinquency = max(
        1 if num_30_59_days_late > 0 else 0,
        2 if num_60_89_days_late > 0 else 0,
        3 if num_90_days_late > 0 else 0
    )

    # Income/debt ratios (Phase 5)
    income_per_dependent = monthly_income_value / (num_dependents + 1)
    debt_to_income_proxy = debt_ratio * monthly_income_value if monthly_income_value > 0 else debt_ratio
    is_zero_income = 1 if monthly_income_value == 0 else 0

    # Log transforms (Phase 5) — using log1p to safely handle zero values
    debt_ratio_log = np.log1p(debt_ratio)
    debt_to_income_proxy_log = np.log1p(debt_to_income_proxy)
    monthly_income_log = np.log1p(monthly_income_value)
    revolving_utilization_log = np.log1p(revolving_utilization)

    # Age bins (Phase 5) — one-hot, <25 is reference/dropped category
    age_25_34 = 1 if 25 <= age <= 34 else 0
    age_35_44 = 1 if 35 <= age <= 44 else 0
    age_45_54 = 1 if 45 <= age <= 54 else 0
    age_55_64 = 1 if 55 <= age <= 64 else 0
    age_65_plus = 1 if age >= 65 else 0

    row = {
        'RevolvingUtilizationOfUnsecuredLines': revolving_utilization,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': num_30_59_days_late,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income_value,
        'NumberOfOpenCreditLinesAndLoans': num_open_credit_lines,
        'NumberOfTimes90DaysLate': num_90_days_late,
        'NumberRealEstateLoansOrLines': num_real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': num_60_89_days_late,
        'NumberOfDependents': num_dependents,
        'MonthlyIncome_missing': monthly_income_missing,
        'WasFlagged_9698': 0,  # new applicants never carry the historical placeholder-code flag
        'TotalPastDue': total_past_due,
        'EverDelinquent': ever_delinquent,
        'WorstDelinquency': worst_delinquency,
        'IncomePerDependent': income_per_dependent,
        'DebtToIncomeProxy': debt_to_income_proxy,
        'IsZeroIncome': is_zero_income,
        'DebtRatio_log': debt_ratio_log,
        'DebtToIncomeProxy_log': debt_to_income_proxy_log,
        'MonthlyIncome_log': monthly_income_log,
        'RevolvingUtilizationOfUnsecuredLines_log': revolving_utilization_log,
        'Age_25-34': age_25_34,
        'Age_35-44': age_35_44,
        'Age_45-54': age_45_54,
        'Age_55-64': age_55_64,
        'Age_65+': age_65_plus,
    }

    FEATURE_COLUMNS = [
        'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
        'MonthlyIncome_missing', 'WasFlagged_9698', 'TotalPastDue', 'EverDelinquent', 'WorstDelinquency',
        'IncomePerDependent', 'DebtToIncomeProxy', 'IsZeroIncome', 'DebtRatio_log',
        'DebtToIncomeProxy_log', 'MonthlyIncome_log', 'RevolvingUtilizationOfUnsecuredLines_log',
        'Age_25-34', 'Age_35-44', 'Age_45-54', 'Age_55-64', 'Age_65+'
    ]
    return pd.DataFrame([row])[FEATURE_COLUMNS]  # enforce exact column order
