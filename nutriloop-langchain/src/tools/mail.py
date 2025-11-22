from langchain_core.tools import Tool

from typing import List
from langchain_core.tools import Tool


def send_email_to_doctor(message: str, doctor_email: str) -> str:
    """Simulate sending an email to the doctor."""
    log = f"[EMAIL → DOCTOR {doctor_email}] {message}"
    print(log)
    return log


def send_email_to_patient(message: str, patient_email: str) -> str:
    """Simulate sending an email to the patient."""
    log = f"[EMAIL → PATIENT {patient_email}] {message}"
    print(log)
    return log


def create_mail_tools(doctor_email: str, patient_email: str) -> List[Tool]:
    """Create email tools for LangChain agents."""
    return [
        Tool(
            name="send_email_to_doctor",
            func=lambda message: send_email_to_doctor(message, doctor_email),
            description="Send an email message to the doctor."
        ),
        Tool(
            name="send_email_to_patient",
            func=lambda message: send_email_to_patient(message, patient_email),
            description="Send an email message to the patient."
        ),
    ]