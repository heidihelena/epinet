"""Governance and consent gate for federated egress.

Nothing in EpiNet's federated path (epinet_federated) ships data on its own —
this module is the gate every aggregate must pass through *before* it is allowed
to leave a site. It fails closed: if any check does not pass, no egress happens.

What this gate ENFORCES in code (technical guarantees):
- consent present, unexpired, and covering the tier being shared;
- a tier ceiling — identifying data (Tier "identifiable") can never cross, and
  the policy caps how high a tier may;
- small-cell suppression — per-class cells below ``min_cell`` are dropped and an
  aggregate over fewer than ``min_cell`` records is refused (statistical
  disclosure control);
- no identifying field names in the payload;
- a disclosure manifest — exactly what crosses, hashed — so egress is *disclosed*;
- an append-only, hash-chained audit ledger of every egress event.

What this gate CANNOT validate (policy / legal — present but not judged here):
- the lawful basis (for EU health data, a GDPR Art. 9 special-category
  condition), the DPIA, the data-sharing agreement / controllership
  determination, the conflict-of-interest declaration, and the consent text.
  The gate REQUIRES these as non-empty, dated references and refuses to run
  without them — but it does not and cannot assess their legal validity. That is
  a DPO / counsel responsibility. This is a guard rail, not legal advice.

Honesty note: the audit ledger is tamper-EVIDENT (a hash chain reveals edits),
not forgery-PROOF (it is not signed or anchored to an external timestamp). Treat
it as an integrity check, not a non-repudiation guarantee.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

# Sensitivity tiers, lowest to highest. Egress is capped at a tier; identifying
# data is never permitted to cross regardless of consent.
TIER_ORDER = {"aggregate": 0, "derived": 1, "identifiable": 2}

# Field-name fragments that must never appear in an egress payload. Aggregates
# carry none of these by construction; the scan is a backstop against a mis-built
# message accidentally smuggling an identifier.
DENY_FIELD_FRAGMENTS = (
    "name", "dob", "birth", "mrn", "ssn", "social", "address", "phone",
    "email", "postcode", "zip", "nhs", "note", "free_text", "comment", "patient_id",
)


class GovernanceError(Exception):
    """Raised when an egress is refused. The gate fails closed."""


def _tier_rank(tier: str) -> int:
    if tier not in TIER_ORDER:
        raise GovernanceError(f"unknown data tier: {tier!r}")
    return TIER_ORDER[tier]


@dataclass
class Consent:
    """A site's recorded authorization to disclose. Policy fields are required
    (non-empty) but their legal validity is not assessed here."""

    site: str
    controller: str            # data controller — the controllership wall
    lawful_basis: str          # e.g. "GDPR Art 9(2)(j) — scientific research"
    dpia_reference: str        # reference to a completed DPIA
    purpose: str               # the disclosed purpose of the egress
    version: str               # consent version
    allowed_tier: str = "aggregate"
    coi_acknowledged: bool = False   # e.g. registry-lead/vendor COI acknowledged
    expires: str | None = None       # ISO date; None = no expiry (discouraged)

    def validate(self, *, tier: str, now: date | None = None) -> None:
        now = now or datetime.now(timezone.utc).date()
        required = {
            "site": self.site, "controller": self.controller,
            "lawful_basis": self.lawful_basis, "dpia_reference": self.dpia_reference,
            "purpose": self.purpose, "version": self.version,
        }
        missing = [k for k, v in required.items() if not (v and str(v).strip())]
        if missing:
            raise GovernanceError(f"consent missing required fields: {missing}")
        if not self.coi_acknowledged:
            raise GovernanceError("conflict-of-interest not acknowledged (coi_acknowledged=False)")
        if self.expires is not None:
            try:
                expiry = date.fromisoformat(self.expires)
            except (ValueError, TypeError) as exc:
                # A malformed expiry must refuse (fail closed) with the documented
                # error type, not crash with a bare ValueError.
                raise GovernanceError(f"consent expiry {self.expires!r} is not an ISO date") from exc
            if expiry < now:
                raise GovernanceError(f"consent expired on {self.expires}")
        if _tier_rank(tier) > _tier_rank(self.allowed_tier):
            raise GovernanceError(
                f"tier {tier!r} exceeds consent allowance {self.allowed_tier!r}"
            )


@dataclass
class DisclosurePolicy:
    """Coordinator-side limits applied on top of consent."""

    min_cell: int = 10                 # small-cell suppression threshold
    allowed_tier: str = "aggregate"    # policy ceiling on egress tier
    deny_fragments: tuple[str, ...] = DENY_FIELD_FRAGMENTS


def _scan_for_identifiers(payload: object, deny: tuple[str, ...], path: str = "") -> list[str]:
    """Return paths where a denied identifier fragment appears (key or string)."""
    hits: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if any(frag in str(key).lower() for frag in deny):
                hits.append(f"{path}{key}")
            hits.extend(_scan_for_identifiers(value, deny, f"{path}{key}."))
    elif isinstance(payload, (list, tuple)):
        for i, value in enumerate(payload):
            if isinstance(value, str) and any(frag in value.lower() for frag in deny):
                hits.append(f"{path}[{i}]={value}")
            hits.extend(_scan_for_identifiers(value, deny, f"{path}[{i}]."))
    return hits


def suppress_small_cells(payload: dict, min_cell: int) -> tuple[dict, list[str]]:
    """Drop per-class cells below ``min_cell`` from a known aggregate shape.

    Handles the EpiNet message shapes: ``class_n``/``class_sum`` (federated fit)
    and ``runner_up_counts`` (federated contestability). Returns a redacted copy
    and the list of suppressed cell labels.
    """
    redacted = json.loads(json.dumps(payload))  # deep copy of JSON-able content
    suppressed: list[str] = []

    if isinstance(redacted.get("class_n"), dict):
        for cls, n in list(redacted["class_n"].items()):
            if int(n) < min_cell:
                redacted["class_n"].pop(cls, None)
                if isinstance(redacted.get("class_sum"), dict):
                    redacted["class_sum"].pop(cls, None)
                suppressed.append(f"class_n[{cls}]={n}")

    if isinstance(redacted.get("runner_up_counts"), dict):
        for cls, n in list(redacted["runner_up_counts"].items()):
            if int(n) < min_cell:
                redacted["runner_up_counts"].pop(cls, None)
                suppressed.append(f"runner_up_counts[{cls}]={n}")

    # Secondary (complementary) suppression: a published total that equals the
    # sum of a group's cells lets a suppressed cell be recovered by subtraction
    # (total - sum(retained)). Where we suppressed any cell in a group, reduce
    # the matching total(s) to the retained sum so nothing is recoverable.
    def _block_subtraction(group_key: str, total_keys: tuple[str, ...]) -> None:
        original = payload.get(group_key)
        if not isinstance(original, dict):
            return
        orig_sum = sum(int(v) for v in original.values())
        retained_sum = sum(int(v) for v in redacted.get(group_key, {}).values())
        if retained_sum == orig_sum:
            return  # nothing suppressed in this group
        for tk in total_keys:
            if isinstance(redacted.get(tk), (int, float)) and int(redacted[tk]) == orig_sum:
                redacted[tk] = retained_sum
                suppressed.append(f"{tk} reduced {orig_sum}->{retained_sum} (block subtraction)")

    _block_subtraction("class_n", ("n", "labeled_count"))
    _block_subtraction("runner_up_counts", ("flip_count", "n_scored", "n"))

    return redacted, suppressed


def _record_count(payload: dict) -> int:
    """Best-effort total-record count from a known aggregate shape."""
    for key in ("n", "n_scored", "flip_count", "labeled_count"):
        if isinstance(payload.get(key), (int, float)):
            return int(payload[key])
    return 0


def _sha256_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class AuditLedger:
    """Append-only, hash-chained log of egress events. Tamper-evident, not
    forgery-proof (unsigned). ``verify`` recomputes the chain."""

    entries: list[dict] = field(default_factory=list)

    def append(self, event: dict, *, timestamp: str | None = None) -> dict:
        prev_hash = self.entries[-1]["entry_hash"] if self.entries else "0" * 64
        body = {
            "seq": len(self.entries),
            "prev_hash": prev_hash,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
        }
        entry = {**body, "entry_hash": _sha256_json(body)}
        self.entries.append(entry)
        return entry

    def verify(self) -> bool:
        prev_hash = "0" * 64
        for entry in self.entries:
            body = {k: entry[k] for k in ("seq", "prev_hash", "timestamp", "event")}
            if entry["prev_hash"] != prev_hash or entry["entry_hash"] != _sha256_json(body):
                return False
            prev_hash = entry["entry_hash"]
        return True


def check_egress(
    payload: dict,
    *,
    policy: DisclosurePolicy,
    consent: Consent,
    tier: str = "aggregate",
    audit: AuditLedger | None = None,
    now: date | None = None,
    timestamp: str | None = None,
) -> tuple[dict, dict]:
    """The gate. Returns (redacted_payload, disclosure_manifest) or raises.

    Order: identifying data is hard-blocked; consent + policy tier ceiling are
    checked; identifier field names are scanned; small cells are suppressed; the
    record floor is enforced; a disclosure manifest is built and (if an audit
    ledger is given) recorded. Fails closed at every step.
    """
    if tier == "identifiable":
        raise GovernanceError("identifying data is never permitted to cross a site boundary")
    if _tier_rank(tier) > _tier_rank(policy.allowed_tier):
        raise GovernanceError(f"tier {tier!r} exceeds policy ceiling {policy.allowed_tier!r}")

    consent.validate(tier=tier, now=now)

    hits = _scan_for_identifiers(payload, policy.deny_fragments)
    if hits:
        raise GovernanceError(f"payload contains identifying fields: {hits[:5]}")

    # Enforce the record floor on the TRUE total (before secondary suppression
    # reduces the disclosed total), so neutralising a leak-prone total cannot
    # also weaken the floor check.
    total_n = _record_count(payload)
    if total_n and total_n < policy.min_cell:
        raise GovernanceError(
            f"aggregate covers {total_n} records (< min_cell={policy.min_cell}); refused"
        )

    redacted, suppressed = suppress_small_cells(payload, policy.min_cell)

    manifest = {
        "tier": tier,
        "fields_disclosed": sorted(redacted.keys()),
        "record_count": _record_count(redacted),
        "suppressed_cells": suppressed,
        "min_cell": policy.min_cell,
        "controller": consent.controller,
        "purpose": consent.purpose,
        "lawful_basis": consent.lawful_basis,
        "dpia_reference": consent.dpia_reference,
        "consent_version": consent.version,
        "payload_sha256": _sha256_json(redacted),
    }
    if audit is not None:
        audit.append({"action": "egress", "manifest": manifest}, timestamp=timestamp)
    return redacted, manifest
