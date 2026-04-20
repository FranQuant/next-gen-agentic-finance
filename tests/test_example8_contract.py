import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import example8


def _packet(labels: tuple[str, ...]) -> str:
    return "\n".join(f"- {label}: value" for label in labels)


def _handoff(evidence: str, research_read: str, open_questions: str, title_2: str = "OPEN QUESTIONS / GAPS") -> str:
    return "\n\n".join(
        [
            "EVIDENCE",
            evidence,
            "RESEARCH READ",
            research_read,
            title_2,
            open_questions,
        ]
    )


def test_validate_bulleted_packet_accepts_exact_labels():
    result = example8.validate_bulleted_packet(_packet(example8.EVIDENCE_LABELS), example8.EVIDENCE_LABELS)

    assert result["ok"] is True
    assert result["missing_labels"] == []
    assert result["unexpected_labels"] == []
    assert result["duplicate_labels"] == []


def test_validate_bulleted_packet_rejects_missing_label():
    labels = example8.EVIDENCE_LABELS[:-1]
    result = example8.validate_bulleted_packet(_packet(labels), example8.EVIDENCE_LABELS)

    assert result["ok"] is False
    assert "Recent Company Catalysts" in result["missing_labels"]
    assert result["present_labels"] == list(labels)


def test_validate_final_handoff_accepts_exact_structure():
    evidence = _packet(example8.EVIDENCE_LABELS)
    research_read = _packet(example8.RESEARCH_READ_LABELS)
    open_questions = _packet(example8.OPEN_QUESTIONS_LABELS)

    result = example8.validate_final_handoff(_handoff(evidence, research_read, open_questions))

    assert result["ok"] is True
    assert result["missing_section_titles"] == []
    assert result["duplicate_section_titles"] == []
    assert result["section_sequence_ok"] is True
    assert all(section["bullet_validation"]["ok"] for section in result["section_diagnostics"].values())


def test_validate_final_handoff_rejects_missing_section_title():
    evidence = _packet(example8.EVIDENCE_LABELS)
    research_read = _packet(example8.RESEARCH_READ_LABELS)
    open_questions = _packet(example8.OPEN_QUESTIONS_LABELS)

    result = example8.validate_final_handoff(_handoff(evidence, research_read, open_questions, title_2="OPEN QUESTIONS"))

    assert result["ok"] is False
    assert "OPEN QUESTIONS / GAPS" in result["missing_section_titles"]
    assert result["section_sequence_ok"] is False
