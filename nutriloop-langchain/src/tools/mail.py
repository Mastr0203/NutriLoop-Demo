from langchain.tools import Tool
from langchain.email import EmailClient

def send_email_to_doctor(message: str, doctor_email: str) -> None:
    """Send an email to the doctor."""
    email_client = EmailClient()
    email_client.send_email(to=doctor_email, subject="Patient Update", body=message)

def send_email_to_patient(message: str, patient_email: str) -> None:
    """Send an email to the patient."""
    email_client = EmailClient()
    email_client.send_email(to=patient_email, subject="Your Meal Plan", body=message)

def create_mail_tools(doctor_email: str, patient_email: str) -> List[Tool]:
    """Create tools for sending emails."""
    return [
        Tool(
            name="send_email_to_doctor",
            func=lambda message: send_email_to_doctor(message, doctor_email),
            description="Send an email to the doctor with the provided message."
        ),
        Tool(
            name="send_email_to_patient",
            func=lambda message: send_email_to_patient(message, patient_email),
            description="Send an email to the patient with the provided message."
        )
    ]