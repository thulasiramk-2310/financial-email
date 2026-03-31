from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from models import LabelType, RiskType, DecisionType
from grader import grade

SPAM_KEYWORDS   = {"won","gift","prize","claim","lottery","guaranteed","double","free","offer","selected","congratulations","pre-approved","limited time"}
URGENT_KEYWORDS = {"urgent","alert","immediate","action required","down","failing","unauthorized","security","verify","critical","breach","suspend"}
HIGH_RISK_WORDS = {"wire transfer","overseas","unauthorized","ssn","bank details","phishing","fraud","critical","production down","security alert"}
MEDIUM_RISK_WORDS = {"overdue","due","compliance","kyc","audit","re-submit","invoice","legal action","approved loan","application"}

def _text(email: dict) -> str:
    return f"{email.get('subject','')} {email.get('content','')} {email.get('sender','')}".lower()

def classify(email: dict) -> str:
    t = _text(email)
    if any(k in t for k in SPAM_KEYWORDS):   return "spam"
    if any(k in t for k in URGENT_KEYWORDS): return "urgent"
    return "finance"

def assess_risk(email: dict, cls: str) -> str:
    if cls == "spam": return "high"
    t = _text(email)
    if any(k in t for k in HIGH_RISK_WORDS):   return "high"
    if any(k in t for k in MEDIUM_RISK_WORDS): return "medium"
    return "low"

def decide(cls: str, risk: str) -> str:
    if cls == "spam":  return "block"
    if risk == "high": return "escalate"
    if risk == "medium": return "review"
    return "pass"

def predict(email: dict, task_name: str = "hard") -> dict:
    cls  = classify(email)
    risk = assess_risk(email, cls)
    dec  = decide(cls, risk)
    result: dict = {"classification": cls}
    if task_name in {"medium", "hard"}: result["risk_level"] = risk
    if task_name == "hard":             result["decision"]   = dec
    return result

def run_inference(emails: list[dict], task_name: str = "hard") -> dict:
    history = []
    for email in emails:
        pred = predict(email, task_name)
        history.append({
            "classification_correct": pred.get("classification") == email.get("true_label"),
            "risk_correct":           pred.get("risk_level")     == email.get("true_risk"),
            "decision_correct":       pred.get("decision")       == email.get("correct_decision"),
            "predicted_risk_level":   pred.get("risk_level"),
            "effective_risk":         email.get("true_risk"),
            "predicted_decision":     pred.get("decision"),
        })
    return grade(task_name, history)

if __name__ == "__main__":
    import json
    test = [
        {"subject":"Wire Transfer $47,500","sender":"alerts@bank.com","content":"overseas account transfer","true_label":"urgent","true_risk":"high","correct_decision":"escalate"},
        {"subject":"You won a gift card!","sender":"spam@promo.com","content":"claim your free reward","true_label":"spam","true_risk":"high","correct_decision":"block"},
        {"subject":"Salary Credited","sender":"hr@company.com","content":"salary processed successfully","true_label":"finance","true_risk":"low","correct_decision":"pass"},
    ]
    print(json.dumps(run_inference(test, "hard"), indent=2))
