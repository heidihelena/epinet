# Governance and consent for federated EpiNet

This document accompanies `epinet_governance.py`. It states plainly **what the
code enforces** and **what remains a policy/legal responsibility the code cannot
discharge**. The goal is a guard rail, not governance theatre — and not legal
advice. Review everything below with your Data Protection Officer and counsel
before any real registry data is involved.

## What the code enforces (technical guarantees)

Every federated aggregate must pass `check_egress()` before it leaves a site.
The gate **fails closed** — if any check does not pass, nothing is disclosed:

| Guarantee | How |
| --- | --- |
| No identifying data crosses | Tier `identifiable` is hard-blocked; a field-name scan rejects payloads carrying identifiers |
| Small-cell suppression | Per-class cells below `min_cell` are dropped; an aggregate over fewer than `min_cell` records is refused (statistical disclosure control) |
| Tier ceiling | Egress is capped at the lower of the consent's and the policy's permitted tier |
| Consent present and current | Required consent fields must be non-empty; expired consent is refused; the conflict-of-interest flag must be acknowledged |
| Disclosed egress | A disclosure manifest records exactly what crosses (fields, record count, suppressions, purpose, lawful basis, content hash) |
| Tamper-evident audit | Every egress is appended to a hash-chained ledger; `verify()` detects edits |

Honesty note: the audit ledger is tamper-**evident** (a hash chain reveals
edits), not forgery-**proof** — it is not signed or externally anchored. Treat it
as an integrity check, not non-repudiation.

## What the code requires but cannot validate (policy / legal)

The gate refuses to run unless these are **present** as non-empty, dated
references. It does **not**, and cannot, assess their legal validity:

- [ ] **Lawful basis.** For EU/EEA health data this is special-category data —
  you need both an Art. 6 basis and an Art. 9 condition (commonly Art. 9(2)(j),
  scientific research, with the safeguards of Art. 89 and applicable national
  law). Finland: check the Secondary Use Act and Findata where relevant.
- [ ] **DPIA.** A completed Data Protection Impact Assessment, referenced by id.
- [ ] **Data-sharing agreement / controllership determination.** Who is
  controller vs processor at each tier (see below).
- [ ] **Consent text / legal basis for the source records**, as required by the
  registry's own governance.
- [ ] **Conflict-of-interest declaration.** Where the same person is both
  registry clinical lead and tool vendor, this must be declared and managed; the
  gate requires it to be acknowledged but cannot judge that the management is
  adequate.

## Two-tier controllership

Federation creates two boundaries, and the controllership question applies at
**both**:

1. **Tier 1 (registry shard → canonical schema).** Each site is controller of
   its own records; the adapter is a processor that only re-formats.
2. **Tier 2 (derived federated dataset).** When aggregates are pooled, determine
   who is controller of the *derived* dataset and the corpus it may feed. Pooling
   is an act with a controller — name them.

Record the controller per site in the `Consent` (`controller` field); do not
merge across controllers without a data-sharing agreement reference.

## "De-identified", not "anonymous"

The aggregates this system emits are **de-identified**, not anonymous.
Re-identification risk is reduced (small-cell suppression, aggregates only) but
not provably zero. Use the word "de-identified" in all documentation and consent
text. Claiming "anonymous" makes a stronger legal assertion than the technique
supports.

## Contribution is active and opt-in

Feeding the derived dataset upward (e.g. to a shared corpus) is a separate,
**active, default-off** choice per site — never automatic. The federated fit and
any contribution are distinct authorizations; consent for one is not consent for
the other. Raw record-level or free-text content is a higher tier and needs its
own, separate consent — never bundled with the aggregate authorization.

---

*This document and `epinet_governance.py` are guard rails for a research
demonstrator. They do not constitute legal advice and do not by themselves
establish compliance with the GDPR, the EU AI Act, the MDR, or national law.*
