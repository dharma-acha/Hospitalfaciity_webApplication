import streamlit as st
import pandas as pd
from joblib import load
import pickle
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('preprocessed_dataset_xdata.csv')
data = pd.read_csv('preprocessed_dataset_new.csv') 
with open("dt.pkl",'rb') as file:
    model = pickle.load(file)

cols = ['Patients Currently Hospitalized', 'Patients Admitted Due to COVID', 'Patients Admitted Not Due to COVID',
        'Patients Newly Admitted', 'Patients Positive After Admission', 'Patients Discharged', 'Patients Currently in ICU',
        'Patients Expired', 'Total Staffed Beds', 'Total Staffed Beds Currently Available', 'Total Staffed ICU Beds',
        'Total Staffed ICU Beds Currently Available', 'Total New Admissions Reported', 'Young', 'Adult', 'Senior',
        'Facility Network_ALBANY MEDICAL CENTER', 'Facility Network_ALLEGHENY HEALTH NETWORK', 'Facility Network_ARNOT HEALTH',
        'Facility Network_ASCENSION HEALTH', 'Facility Network_BASSETT HEALTHCARE NETWORK',
        'Facility Network_CATHOLIC HEALTH SERVICES OF LONG ISLAND', 'Facility Network_CATHOLIC HEALTH, BUFFALO',
        'Facility Network_CAYUGA HEALTH SYSTEM', 'Facility Network_CROUSE HEALTH', 'Facility Network_FINGER LAKES HEALTH',
        'Facility Network_GARNET HEALTH -FORMERLY GREATER HUDSON VALLEY HEALTH SYSTEM', 'Facility Network_INDEPENDENT',
        'Facility Network_KALEIDA HEALTH', 'Facility Network_MEDISYS HEALTH NETWORK', 'Facility Network_MOHAWK VALLEY HEALTH SYSTEM',
        'Facility Network_MONTEFIORE HEALTHCARE SYSTEM', 'Facility Network_MOUNT SINAI HEALTH SYSTEM',
        'Facility Network_NEW YORK-PRESBYTERIAN HEALTHCARE SYSTEM', 'Facility Network_NORTH STAR HEALTH ALLIANCE',
        'Facility Network_NORTHWELL HEALTH', 'Facility Network_NUVANCE HEALTH', 'Facility Network_NYC H+H',
        'Facility Network_NYU LANGONE HEALTH', 'Facility Network_ONE BROOKLYN HEALTH SYSTEM',
        'Facility Network_RIVERSIDE HEALTH CARE SYSTEM, INC.', 'Facility Network_ROCHESTER REGIONAL HEALTH SYSTEM',
        'Facility Network_ST. JOSEPHS HOSPITAL SYRACUSE NY', 'Facility Network_ST. LAWRENCE HEALTH SYSTEM',
        'Facility Network_ST. PETERS HEALTH PARTNERS', 'Facility Network_STONY BROOK MEDICINE', 'Facility Network_THE GUTHRIE CLINIC',
        'Facility Network_THE UNIVERSITY OF VERMONT HEALTH NETWORK', 'Facility Network_THE UNIVERSITY OF VERMONT HEALTH NETWORK ELIZABETH',
        'Facility Network_TRINITY', 'Facility Network_UNITED HEALTH SERVICES HOSPITALS, INC.',
        'Facility Network_UNIVERSITY OF ROCHESTER MEDICAL CENTER', 'Facility Network_UNIVERSITY OF VERMONT HEALTH NETWORK',
        'Facility Network_UPMC', 'Facility Network_WESTCHESTER MEDICAL CENTER HEALTH NETWORK']


def preprocess_new_input(input_data):
    
    input_data_processed = pd.get_dummies(input_data, columns=['Facility Name', 'Facility County', 'Facility Network'])
    
    
    for col in df.columns:
        if col not in input_data_processed.columns:
            input_data_processed[col] = 0  
    
    
    scaler = StandardScaler().fit(df.select_dtypes(include=[float]))  
    numerical_cols = input_data_processed.select_dtypes(include=[float]).columns
    input_data_processed[numerical_cols] = scaler.transform(input_data_processed[numerical_cols])

    input_data_processed = input_data_processed.reindex(columns=cols, fill_value=0)  
    return input_data_processed


st.title('Hospital Facility Data Preprocessor')


with st.form("new_data_form", clear_on_submit=True):
    st.write("Enter new facility data:")
    facility_name = st.selectbox("Facility Name", options=data['Facility Name'].unique())
    facility_county = st.selectbox("Facility County", options=data['Facility County'].unique())
    facility_network = st.selectbox("Facility Network", options=data['Facility Network'].unique())


    patients_currently_hospitalized = st.text_input("Patients Currently Hospitalized", value="0")
    patients_admitted_due_to_covid = st.text_input("Patients Admitted Due to COVID", value="0")
    patients_admitted_not_due_to_covid = st.text_input("Patients Admitted Not Due to COVID", value="0")
    patients_newly_admitted = st.text_input("Patients Newly Admitted", value="0")
    patients_positive_after_admission = st.text_input("Patients Positive After Admission", value="0")
    patients_discharged = st.text_input("Patients Discharged", value="0")
    patients_currently_in_icu = st.text_input("Patients Currently in ICU", value="0")
    patients_expired = st.text_input("Patients Expired", value="0")
    total_staffed_beds = st.text_input("Total Staffed Beds", value="0")
    total_staffed_beds_currently_available = st.text_input("Total Staffed Beds Currently Available", value="0")
    total_staffed_icu_beds = st.text_input("Total Staffed ICU Beds", value="0")
    total_staffed_icu_beds_currently_available = st.text_input("Total Staffed ICU Beds Currently Available", value="0")
    total_new_admissions_reported = st.text_input("Total New Admissions Reported", value="0")
    young = st.text_input("Young", value="0")
    adult = st.text_input("Adult", value="0")
    senior = st.text_input("Senior", value="0")

    submitted = st.form_submit_button("Submit")
    if submitted:
        new_data = pd.DataFrame({
            'Facility Name': [facility_name],
            'Facility County': [facility_county],
            'Facility Network': [facility_network],
            'Patients Currently Hospitalized': [int(patients_currently_hospitalized)],
            'Patients Admitted Due to COVID': [int(patients_admitted_due_to_covid)],
            'Patients Admitted Not Due to COVID': [int(patients_admitted_not_due_to_covid)],
            'Patients Newly Admitted': [int(patients_newly_admitted)],
            'Patients Positive After Admission': [int(patients_positive_after_admission)],
            'Patients Discharged': [int(patients_discharged)],
            'Patients Currently in ICU': [int(patients_currently_in_icu)],
            'Patients Expired': [int(patients_expired)],
            'Total Staffed Beds': [int(total_staffed_beds)],
            'Total Staffed Beds Currently Available': [int(total_staffed_beds_currently_available)],
            'Total Staffed ICU Beds': [int(total_staffed_icu_beds)],
            'Total Staffed ICU Beds Currently Available': [int(total_staffed_icu_beds_currently_available)],
            'Total New Admissions Reported': [int(total_new_admissions_reported)],
            'Young': [int(young)],
            'Adult': [int(adult)],
            'Senior': [int(senior)]
        })

        st.write("Submitted data:", new_data)
        
        
        preprocessed_input = preprocess_new_input(new_data)

        pred = model.predict(preprocessed_input)
        

        st.write("Preprocessed input data:")
        st.dataframe(preprocessed_input)
