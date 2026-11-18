# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local mock ledger, policy check, and structured logging tools for expenses."""

from __future__ import annotations

import json
from typing import Any

APPROVAL_THRESHOLD_USD = 100.0

_CATEGORY_GUIDANCE: dict[str, str] = {
    "meals": "Itemized receipt and attendee list required for meals >= $100.",
    "travel": "Airfare and rail bookings >= $100 require trip business justification.",
    "lodging": "Hotel stays require nightly rate breakdown and folio receipt.",
    "software": "Software subscriptions >= $100 require manager and IT license check.",
    "equipment": "Hardware/equipment >= $100 requires asset tagging upon approval.",
    "supplies": "Office supplies >= $100 require itemized vendor invoice.",
}

_EXPENSE_LEDGER: list[dict[str, Any]] = []


def clear_expense_ledger() -> None:
    """Clear the in-memory expense ledger (used for deterministic test isolation)."""
    _EXPENSE_LEDGER.clear()


def check_expense_policy(amount: float, category: str) -> dict[str, Any]:
    """Evaluate company expense policy rules and risk indicators for a claim.

    Args:
        amount: Expense claim amount in USD.
        category: Expense category such as meals, travel, lodging, or software.

    Returns:
        Dictionary containing policy threshold info, risk flags, and guidance.
    """
    normalized_category = (category or "other").strip().lower()
    flags: list[str] = []

    if amount >= 1000.0:
        flags.append("high_value_over_1000")
    if (
        amount >= APPROVAL_THRESHOLD_USD
        and float(amount).is_integer()
        and int(amount) % 100 == 0
    ):
        flags.append("round_dollar_amount")
    if normalized_category not in _CATEGORY_GUIDANCE:
        flags.append("uncategorized_or_other")

    guidance = _CATEGORY_GUIDANCE.get(
        normalized_category,
        "Verify itemized receipt and business purpose before approving.",
    )
    risk_level = (
        "high"
        if amount >= 1000.0
        else ("medium" if flags or amount >= 250.0 else "low")
    )
    return {
        "threshold_usd": APPROVAL_THRESHOLD_USD,
        "requires_human_review": amount >= APPROVAL_THRESHOLD_USD,
        "category": normalized_category,
        "risk_level": risk_level,
        "risk_flags": flags,
        "policy_guidance": guidance,
    }


def emit_expense_alert(
    submitter: str,
    amount: float,
    category: str,
    risk_summary: str,
) -> dict[str, Any]:
    """Emit a structured JSON warning log alerting reviewers of a >= $100 claim.

    Args:
        submitter: Employee identifier or email who submitted the expense.
        amount: Expense amount in USD.
        category: Expense category.
        risk_summary: Brief summary explaining why human review is required.

    Returns:
        Confirmation dictionary indicating the alert was logged.
    """
    log_entry = {
        "severity": "WARNING",
        "alert_type": "expense_review",
        "message": f"Expense review alert: ${amount:.2f} from {submitter} — {risk_summary}",
        "submitter": submitter,
        "amount": amount,
        "category": category,
        "risk_summary": risk_summary,
    }
    print(json.dumps(log_entry), flush=True)
    return {"status": "alert_emitted", "submitter": submitter, "amount": amount}


def record_expense_decision(
    expense_id: str,
    employee: str,
    amount: float,
    category: str,
    description: str,
    status: str,
    decided_by: str,
    reason: str,
) -> dict[str, Any]:
    """Record an expense approval or rejection in the local ledger and stdout log.

    Args:
        expense_id: Unique identifier of the expense claim.
        employee: Employee email or name who submitted the claim.
        amount: Expense amount in USD.
        category: Expense category.
        description: Description of the expense.
        status: Decision status ('approved' or 'rejected').
        decided_by: Actor recording the decision ('auto_approve' or 'human_reviewer').
        reason: Explanation of the decision.

    Returns:
        The recorded ledger entry dictionary.
    """
    entry: dict[str, Any] = {
        "expense_id": expense_id,
        "employee": employee,
        "amount": round(float(amount), 2),
        "currency": "USD",
        "category": category,
        "description": description,
        "status": status,
        "decided_by": decided_by,
        "reason": reason,
    }
    _EXPENSE_LEDGER.append(entry)

    log_entry = {
        "severity": "INFO" if status == "approved" else "WARNING",
        "message": (
            f"Expense {expense_id} {status} by {decided_by}: "
            f"${amount:.2f} ({category}) for {employee}"
        ),
        "decision": status,
        **entry,
    }
    print(json.dumps(log_entry), flush=True)
    return entry


def get_expense_ledger() -> dict[str, Any]:
    """Return all recorded expense decisions from the in-memory ledger.

    Returns:
        Dictionary containing ledger entry count and list of entries.
    """
    return {
        "status": "success",
        "count": len(_EXPENSE_LEDGER),
        "entries": list(_EXPENSE_LEDGER),
    }
