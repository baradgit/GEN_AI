import os
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from typing import Union
from langchain_core.pydantic_v1 import BaseModel, Field

# Set the title of the web app based on the problem statement
st.title("Tablet and Doctor Recommendation System")
st.write("Example text 1.patient having fever please suggest tablets and other details.")
st.write("Example text 2.patient having back pain can you suggest which doctor to consult and other details.")

# Prompt user to enter their Groq API Key securely
GROQ_API_KEY = st.text_input("Enter your Groq API Key", type="password")

# Error handling for missing API key
if not GROQ_API_KEY:
    st.error("Please enter your Groq API Key to proceed.")
else:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    # Initialize the language model
    try:
        llm = ChatGroq(model="llama3-8b-8192")
    except Exception as e:
        st.error(f"Failed to initialize the language model: {e}")

    # Define the Pydantic classes
    class TabletInfo(BaseModel):
        Patient_issue: str = Field(description="The issue the patient is facing")
        tablet_name: str = Field(description="The name of the tablet")
        price: float = Field(description="The price of the tablet")
        brand: str = Field(description="The name of the brand")

    class DoctorSupportResponse(BaseModel):
        Patient_issue: str = Field(description="The issue the patient is facing")
        doctor: str = Field(description="Type of doctor to consult")
        consultation_fee: float = Field(description="The consultation fee")

    class Response(BaseModel):
        output: Union[TabletInfo, DoctorSupportResponse]

    # Function to display the response as a table
    def display_response_as_table(response: Response, tablet_placeholder, doctor_placeholder):
        if isinstance(response.output, TabletInfo):
            df = pd.DataFrame([response.output.dict()])
            tablet_placeholder.write("Product Information:")
            tablet_placeholder.table(df)
        elif isinstance(response.output, DoctorSupportResponse):
            df = pd.DataFrame([response.output.dict()])
            doctor_placeholder.write("Doctor Support Response:")
            doctor_placeholder.table(df)
        else:
            st.error("Unknown response type received.")

    # Structure the model to return output based on the schema
    try:
        structured_llm = llm.with_structured_output(Response)
    except Exception as e:
        st.error(f"Error in structuring the language model output: {e}")

    # Display empty tables initially
    tablet_placeholder = st.empty()
    doctor_placeholder = st.empty()

    # Initialize empty dataframes for the placeholders
    empty_tablet_df = pd.DataFrame(columns=["Patient_issue", "tablet_name", "price", "brand"])
    empty_doctor_df = pd.DataFrame(columns=["Patient_issue", "doctor_type", "consultation_fee"])

    # Display empty tables at the start
    tablet_placeholder.write("Product Information:")
    tablet_placeholder.table(empty_tablet_df)

    doctor_placeholder.write("Doctor Support Response:")
    doctor_placeholder.table(empty_doctor_df)

    # Input query section
    input_text = st.text_input("Enter your query")

    # Submit button
    if st.button("Submit"):
        if not input_text:
            st.error("Please enter a query before submitting.")
        else:
            try:
                # Invoke the model and display the result
                response = structured_llm.invoke(input_text)
                display_response_as_table(response, tablet_placeholder, doctor_placeholder)
            except Exception as e:
                st.error(f"Unable to process the query: {e}. Please provide clear and valid information.")
