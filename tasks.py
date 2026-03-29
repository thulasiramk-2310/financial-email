from __future__ import annotations
from models import TaskSpec

TASKS: list[TaskSpec] = [
    TaskSpec(
        name="easy",
        difficulty="easy",
        required_fields=["classification"],
        description=(
            "Classify each financial email into one of three categories: "
            "'spam', 'finance', or 'urgent'. "
            "Spam emails are unsolicited promotional or phishing messages. "
            "Finance emails contain transactional, invoice, or account information. "
            "Urgent emails require immediate attention or action."
        ),
        examples=[
            {
                "email": "Congratulations! You've won a $1000 gift card. Click here to claim now before it expires!",
                "ground_truth": {"classification": "spam"},
            },
            {
                "email": "Your invoice #INV-4521 for $3,200 is attached. Payment is due within 15 days.",
                "ground_truth": {"classification": "finance"},
            },
            {
                "email": "URGENT: Production database is down. All services affected. Immediate action required.",
                "ground_truth": {"classification": "urgent"},
            },
            {
                "email": "Limited time offer! Get 90% off premium subscriptions. Act now — offer expires tonight!",
                "ground_truth": {"classification": "spam"},
            },
            {
                "email": "Your Q3 salary credit of $5,400 has been processed and will reflect in 2 business days.",
                "ground_truth": {"classification": "finance"},
            },
            {
                "email": "Security alert: Unauthorized login attempt detected on your account from an unknown device.",
                "ground_truth": {"classification": "urgent"},
            },
            {
                "email": "Dear customer, your loan application #LN-8823 has been approved. See terms attached.",
                "ground_truth": {"classification": "finance"},
            },
            {
                "email": "You have been selected for a special reward! Reply with your bank details to claim.",
                "ground_truth": {"classification": "spam"},
            },
        ],
    ),
    TaskSpec(
        name="medium",
        difficulty="medium",
        required_fields=["classification", "risk_level"],
        description=(
            "Classify each financial email ('spam', 'finance', 'urgent') AND assign a risk level: "
            "'low', 'medium', or 'high'. "
            "Risk reflects potential financial, reputational, or operational impact. "
            "High risk: phishing, fraud, system failures, large unauthorized transactions. "
            "Medium risk: overdue payments, account anomalies, compliance notices. "
            "Low risk: routine statements, approved transactions, informational updates."
        ),
        examples=[
            {
                "email": "Your credit card ending in 4242 was charged $8,750 by an unknown merchant.",
                "ground_truth": {"classification": "urgent", "risk_level": "high"},
            },
            {
                "email": "Reminder: Your invoice #INV-2201 of $450 is 10 days overdue. Please arrange payment.",
                "ground_truth": {"classification": "finance", "risk_level": "medium"},
            },
            {
                "email": "Your monthly account statement for March is ready. No action required.",
                "ground_truth": {"classification": "finance", "risk_level": "low"},
            },
            {
                "email": "Claim your lottery winnings of $50,000! Send your SSN to verify identity.",
                "ground_truth": {"classification": "spam", "risk_level": "high"},
            },
            {
                "email": "Wire transfer of $125,000 has been initiated from your business account.",
                "ground_truth": {"classification": "urgent", "risk_level": "high"},
            },
            {
                "email": "Your direct deposit of $2,800 has been successfully processed.",
                "ground_truth": {"classification": "finance", "risk_level": "low"},
            },
            {
                "email": "Compliance notice: Your account requires KYC document re-submission within 7 days.",
                "ground_truth": {"classification": "finance", "risk_level": "medium"},
            },
            {
                "email": "Double your investment in 24 hours! Guaranteed returns. Click to invest now.",
                "ground_truth": {"classification": "spam", "risk_level": "high"},
            },
        ],
    ),
    TaskSpec(
        name="hard",
        difficulty="hard",
        required_fields=["classification", "risk_level", "decision"],
        description=(
            "Run the full financial email triage pipeline: "
            "(1) Classify email as 'spam', 'finance', or 'urgent'. "
            "(2) Assign risk level: 'low', 'medium', or 'high'. "
            "(3) Make a decision: 'block' (spam/fraud), 'escalate' (high risk needing human review), "
            "'review' (medium risk, needs attention), or 'pass' (low risk, routine). "
            "Decision logic: spam → block; urgent+high → escalate; finance+medium → review; low risk → pass."
        ),
        examples=[
            {
                "email": "ALERT: $47,500 wire transfer initiated to overseas account. Confirm or deny immediately.",
                "ground_truth": {
                    "classification": "urgent",
                    "risk_level": "high",
                    "decision": "escalate",
                },
            },
            {
                "email": "Congratulations! You've been pre-approved for a $100,000 loan. Provide SSN to proceed.",
                "ground_truth": {
                    "classification": "spam",
                    "risk_level": "high",
                    "decision": "block",
                },
            },
            {
                "email": "Invoice #5543 for $12,000 is 30 days overdue. Legal action may follow.",
                "ground_truth": {
                    "classification": "finance",
                    "risk_level": "medium",
                    "decision": "review",
                },
            },
            {
                "email": "Your scheduled EMI of $320 was successfully debited on 1st March.",
                "ground_truth": {
                    "classification": "finance",
                    "risk_level": "low",
                    "decision": "pass",
                },
            },
            {
                "email": "Critical: Payment gateway is down. All transactions failing. Investigate immediately.",
                "ground_truth": {
                    "classification": "urgent",
                    "risk_level": "high",
                    "decision": "escalate",
                },
            },
            {
                "email": "Win a free iPhone 15! Just complete a quick survey and enter your card details.",
                "ground_truth": {
                    "classification": "spam",
                    "risk_level": "high",
                    "decision": "block",
                },
            },
            {
                "email": "Regulatory audit scheduled for next week. Please prepare Q4 financial records.",
                "ground_truth": {
                    "classification": "finance",
                    "risk_level": "medium",
                    "decision": "review",
                },
            },
            {
                "email": "Your account statement for February 2026 is now available for download.",
                "ground_truth": {
                    "classification": "finance",
                    "risk_level": "low",
                    "decision": "pass",
                },
            },
        ],
    ),
]

TASK_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    task.name: tuple(task.required_fields) for task in TASKS
}


def get_tasks() -> list[dict]:
    return [task.model_dump() for task in TASKS]