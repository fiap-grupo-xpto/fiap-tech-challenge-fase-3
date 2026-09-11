import os
import sqlite3
from datetime import datetime, timedelta


def bootstrap(db_path: str) -> str:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(
            """
            CREATE TABLE patients (
                patient_id TEXT PRIMARY KEY,
                name_synthetic TEXT NOT NULL,
                age INTEGER NOT NULL,
                sex TEXT NOT NULL,
                smoking_history TEXT NOT NULL,
                last_updated_at TEXT NOT NULL
            );

            CREATE TABLE symptoms (
                patient_id TEXT NOT NULL,
                symptom TEXT NOT NULL,
                severity TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            );

            CREATE TABLE pending_exams (
                patient_id TEXT NOT NULL,
                exam_name TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            );

            CREATE TABLE clinical_notes (
                patient_id TEXT NOT NULL,
                note_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            );

            CREATE TABLE latest_findings (
                patient_id TEXT NOT NULL,
                finding TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            );

            CREATE TABLE protocols (
                protocol_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                keywords TEXT NOT NULL,
                content TEXT NOT NULL,
                restricted_actions TEXT NOT NULL
            );

            CREATE TABLE audit_log (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                patient_id TEXT,
                question TEXT,
                status TEXT,
                assistant_answer TEXT,
                sources_used TEXT,
                llm_backend_used TEXT,
                fallback_used INTEGER,
                blocked INTEGER,
                block_reason TEXT,
                requires_human_review INTEGER,
                input_validation TEXT,
                output_validation TEXT,
                error_message TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                attempted_backend TEXT,
                attempted_backend_error TEXT,
                raw_answer TEXT
            );
            """
        )

        now = datetime(2026, 9, 9, 8, 30, 0)
        patients = [
            (
                "P001",
                "Paciente Sintético 001",
                58,
                "M",
                "active smoker",
                (now - timedelta(minutes=15)).isoformat(),
            ),
            (
                "P002",
                "Paciente Sintético 002",
                34,
                "F",
                "never smoker",
                (now - timedelta(hours=2)).isoformat(),
            ),
            (
                "P003",
                "Paciente Sintético 003",
                63,
                "M",
                "former smoker",
                (now - timedelta(minutes=55)).isoformat(),
            ),
            (
                "P004",
                "Paciente Sintético 004",
                71,
                "F",
                "former smoker",
                (now - timedelta(minutes=8)).isoformat(),
            ),
            (
                "P005",
                "Paciente Sintético 005",
                49,
                "M",
                "active smoker",
                (now - timedelta(minutes=25)).isoformat(),
            ),
        ]
        conn.executemany(
            "INSERT INTO patients(patient_id, name_synthetic, age, sex, smoking_history, last_updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            patients,
        )

        symptoms = [
            ("P001", "persistent cough", "moderate"),
            ("P001", "dyspnea", "moderate"),
            ("P001", "weight loss", "mild"),
            ("P002", "intermittent cough", "mild"),
            ("P002", "fatigue", "mild"),
            ("P003", "hemoptysis", "high"),
            ("P003", "chest pain", "moderate"),
            ("P004", "shortness of breath", "moderate"),
            ("P004", "wheezing", "mild"),
            ("P005", "persistent cough", "high"),
            ("P005", "fever", "mild"),
        ]
        conn.executemany(
            "INSERT INTO symptoms(patient_id, symptom, severity) VALUES (?, ?, ?);",
            symptoms,
        )

        pending_exams = [
            ("P001", "chest_ct", "pending", (now - timedelta(days=1)).isoformat()),
            ("P003", "chest_ct", "pending", (now - timedelta(days=2)).isoformat()),
            ("P004", "spirometry", "pending", (now - timedelta(days=3)).isoformat()),
            ("P005", "chest_xray", "pending", (now - timedelta(hours=5)).isoformat()),
        ]
        conn.executemany(
            "INSERT INTO pending_exams(patient_id, exam_name, status, requested_at) VALUES (?, ?, ?, ?);",
            pending_exams,
        )

        notes = [
            (
                "P001",
                "triage",
                "Paciente com tosse persistente e dispneia, histórico de tabagismo ativo.",
                (now - timedelta(minutes=15)).isoformat(),
            ),
            (
                "P002",
                "triage",
                "Paciente com sintomas leves e sem fatores de risco significativos.",
                (now - timedelta(hours=2)).isoformat(),
            ),
            (
                "P003",
                "follow_up",
                "Paciente com hemoptise e dor torácica. Recomenda-se investigação com urgência.",
                (now - timedelta(minutes=55)).isoformat(),
            ),
            (
                "P004",
                "triage",
                "Paciente com dispneia e DPOC conhecida. Necessita avaliar função pulmonar.",
                (now - timedelta(minutes=8)).isoformat(),
            ),
            (
                "P005",
                "triage",
                "Paciente febril com tosse intensa. Considerar causas infecciosas e investigação por imagem.",
                (now - timedelta(minutes=25)).isoformat(),
            ),
        ]
        conn.executemany(
            "INSERT INTO clinical_notes(patient_id, note_type, content, created_at) VALUES (?, ?, ?, ?);",
            notes,
        )

        findings = [
            (
                "P001",
                "High-risk triage indicators based on symptoms and smoking history",
                "alto",
                (now - timedelta(minutes=12)).isoformat(),
            ),
            (
                "P002",
                "Low-risk triage indicators; monitor symptoms",
                "baixo",
                (now - timedelta(hours=2)).isoformat(),
            ),
            (
                "P003",
                "Suspicious symptoms (hemoptysis) requiring escalation",
                "alto",
                (now - timedelta(minutes=50)).isoformat(),
            ),
            (
                "P004",
                "Chronic respiratory disease with moderate acute symptoms",
                "moderado",
                (now - timedelta(minutes=7)).isoformat(),
            ),
            (
                "P005",
                "Potential infection overlap; needs differential assessment",
                "moderado",
                (now - timedelta(minutes=20)).isoformat(),
            ),
        ]
        conn.executemany(
            "INSERT INTO latest_findings(patient_id, finding, risk_level, created_at) VALUES (?, ?, ?, ?);",
            findings,
        )

        protocols = [
            (
                "PROTO-TRIAGE-001",
                "Pulmonary Triage Protocol",
                "triage",
                "cough,dyspnea,smoker,hemoptysis,weight loss,chest pain",
                "When persistent respiratory symptoms coexist with smoking history or hemoptysis, "
                "prioritize clinician review and complete pending imaging before concluding next steps.",
                "do not confirm diagnosis; do not prescribe without clinician validation",
            ),
            (
                "PROTO-EXAMS-001",
                "Pending Exams Follow-up Protocol",
                "exams",
                "pending,exam,ct,xray,spirometry",
                "If an exam is pending, verify status, expected completion date, and whether escalation is required. "
                "Communicate pending exams as an alert for the medical team.",
                "do not override exam ordering; do not fabricate results",
            ),
            (
                "PROTO-ESC-001",
                "Escalation Protocol for Suspicious Findings",
                "escalation",
                "hemoptysis,high-risk,suspicious,urgent",
                "For high-risk indicators such as hemoptysis or high-risk triage, recommend immediate clinician review "
                "and escalation to specialized assessment based on internal triage criteria.",
                "do not claim malignancy; avoid definitive language",
            ),
            (
                "PROTO-HITL-001",
                "Human Validation Requirement",
                "safety",
                "validation,human,clinician,review",
                "All assistant suggestions require clinician validation. The assistant supports decision-making but "
                "must not replace professional judgment.",
                "do not provide final decisions; do not provide final diagnosis",
            ),
            (
                "PROTO-NO-RX-001",
                "No Direct Prescription Rule",
                "safety",
                "prescribe,medication,drug,recipe",
                "The assistant must never prescribe directly. It may suggest that a clinician consider reviewing "
                "treatment options per protocol.",
                "do not prescribe; do not provide dosage instructions",
            ),
        ]
        conn.executemany(
            "INSERT INTO protocols(protocol_id, title, category, keywords, content, restricted_actions) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            protocols,
        )

        conn.commit()
        return db_path
    finally:
        conn.close()


if __name__ == "__main__":
    default_path = os.path.join(os.path.dirname(__file__), "hospital.db")
    bootstrap(default_path)
    print(default_path)

