"""
X12 837P Claim Builder
Generates real ANSI X12 837P (Professional) claim transactions.
This is the actual format used by US healthcare insurers (CMS, BlueCross, Aetna, Cigna, United).

837P Segment Reference:
  ISA  - Interchange Control Header
  GS   - Functional Group Header
  ST   - Transaction Set Header
  BPR  - Beginning of Financial Information
  NM1  - Name (Billing Provider, Subscriber, Patient)
  CLM  - Claim Information
  DTP  - Date/Time Reference
  HI   - Health Care Diagnosis Codes (ICD-10)
  SV1  - Professional Service (CPT code, amount, units)
  SE   - Transaction Set Trailer
  GE   - Functional Group Trailer
  IEA  - Interchange Control Trailer
"""

from datetime import datetime
import textwrap


def build_837p(claim: dict) -> str:
    """
    Build a real X12 837P transaction from a claim dict.

    Required keys:
        patient_id, patient_name, dob, gender,
        insurer, provider_npi, provider_name, provider_tax_id,
        icd_code, cpt_code, claim_amount,
        service_date, prior_auth (optional)
    """
    now = datetime.now()
    date_str  = now.strftime("%Y%m%d")
    time_str  = now.strftime("%H%M")
    isa_ctrl  = now.strftime("%Y%m%d%H%M%S")[:9]   # 9-digit control number

    # ── Parse patient name ──────────────────────────────────────────────
    full_name  = claim.get("patient_name", "UNKNOWN PATIENT")
    parts      = full_name.strip().split()
    last_name  = parts[-1].upper() if parts else "UNKNOWN"
    first_name = parts[0].upper() if len(parts) > 1 else "PATIENT"
    mid_init   = parts[1][0].upper() if len(parts) > 2 else ""

    # ── Claim fields ────────────────────────────────────────────────────
    claim_id      = claim.get("claim_id", f"CLM-{date_str}-{claim.get('patient_id','UNK')}")
    patient_id    = claim.get("patient_id", "UNK")
    dob           = (claim.get("dob") or "19800101").replace("-", "")
    gender        = "M" if str(claim.get("gender","")).upper().startswith("M") else "F"
    icd           = claim.get("icd_code") or claim.get("diagnosis_code", "Z00.00")
    cpt           = claim.get("cpt_code") or claim.get("procedure_code", "99213")
    amount        = f"{float(claim.get('claim_amount', 0)):.2f}"
    svc_date      = (claim.get("service_date") or date_str).replace("-", "")
    prior_auth    = claim.get("prior_auth") or ""
    provider_npi  = claim.get("provider_npi", "1234567890")
    provider_name = (claim.get("provider") or "CITY MEDICAL CENTER").upper()
    provider_tax  = claim.get("provider_tax_id", "123456789")
    insurer       = (claim.get("insurer") or "UNKNOWN").upper()
    member_id     = claim.get("insurance_id") or patient_id
    place_of_svc  = claim.get("place_of_service", "11")   # 11 = Office
    units         = claim.get("units", "1")

    # ── Payer ID mapping (real payer IDs used in production) ────────────
    payer_ids = {
        "BLUECROSS": "00060", "BCBS": "00060",
        "AETNA":     "60054",
        "CIGNA":     "62308",
        "UNITED":    "87726", "UNITEDHEALTHCARE": "87726",
        "MEDICARE":  "00120",
        "MEDICAID":  "77013",
    }
    payer_id = payer_ids.get(insurer.replace(" ", "").upper(), "99999")

    segs = []

    # ISA — Interchange Control Header (fixed 106-char format)
    segs.append(
        f"ISA*00*          *00*          *ZZ*{provider_tax:<15}*ZZ*{payer_id:<15}"
        f"*{date_str[2:]}*{time_str}*^*00501*{isa_ctrl}*0*P*:"
    )

    # GS — Functional Group Header
    segs.append(f"GS*HC*{provider_tax}*{payer_id}*{date_str}*{time_str}*1*X*005010X222A1")

    # ST — Transaction Set Header
    segs.append("ST*837*0001*005010X222A1")

    # BHT — Beginning of Hierarchical Transaction
    segs.append(f"BHT*0019*00*{claim_id}*{date_str}*{time_str}*CH")

    # ── Billing Provider Loop (2000A / 2010AA) ──────────────────────────
    segs.append("HL*1**20*1")
    segs.append("PRV*BI*PXC*207Q00000X")   # Specialty: General Practice
    segs.append(f"NM1*85*2*{provider_name}*****XX*{provider_npi}")
    segs.append(f"N3*123 MEDICAL PLAZA")
    segs.append(f"N4*CHICAGO*IL*60601")
    segs.append(f"REF*EI*{provider_tax}")   # Tax ID

    # ── Subscriber Loop (2000B / 2010BA) ────────────────────────────────
    segs.append("HL*2*1*22*0")
    segs.append("SBR*P*18*******CI")        # P=Primary, 18=Self, CI=Commercial Insurance
    segs.append(f"NM1*IL*1*{last_name}*{first_name}*{mid_init}***MI*{member_id}")
    segs.append(f"N3*{claim.get('address','123 MAIN ST')}")
    segs.append(f"N4*SPRINGFIELD*IL*62701")
    segs.append(f"DMG*D8*{dob}*{gender}")   # Date of Birth + Gender

    # ── Payer (2010BB) ───────────────────────────────────────────────────
    segs.append(f"NM1*PR*2*{insurer}*****PI*{payer_id}")

    # ── Claim Information (2300) ─────────────────────────────────────────
    segs.append(f"CLM*{claim_id}*{amount}***{place_of_svc}:B:1*Y*A*Y*I")
    #            claim_id  total_charge  place_of_svc  signature_on_file  assignment_of_benefits

    # DTP — Service Date
    segs.append(f"DTP*472*D8*{svc_date}")

    # REF — Prior Authorization (if present)
    if prior_auth and str(prior_auth).lower() not in ("false", "none", "not provided", ""):
        segs.append(f"REF*G1*{prior_auth}")

    # HI — Diagnosis Codes (ICD-10-CM)
    # Primary diagnosis
    icd_clean = icd.replace(".", "")
    segs.append(f"HI*ABK:{icd_clean}")     # ABK = ICD-10-CM Principal Diagnosis

    # ── Service Line (2400) ──────────────────────────────────────────────
    segs.append("LX*1")
    segs.append(f"SV1*HC:{cpt}*{amount}*UN*{units}**1")
    #             HC=HCPCS/CPT  charge  UN=units  quantity  diagnosis_pointer
    segs.append(f"DTP*472*D8*{svc_date}")

    # SE — Transaction Set Trailer
    seg_count = len(segs) + 1   # +1 for SE itself
    segs.append(f"SE*{seg_count}*0001")

    # GE — Functional Group Trailer
    segs.append("GE*1*1")

    # IEA — Interchange Control Trailer
    segs.append(f"IEA*1*{isa_ctrl}")

    return "~\n".join(segs) + "~"


def parse_837p_summary(x12_text: str) -> dict:
    """Extract key fields from an 837P for display purposes."""
    summary = {}
    for line in x12_text.replace("~\n", "\n").split("\n"):
        seg = line.strip().rstrip("~")
        els = seg.split("*")
        if not els:
            continue
        sid = els[0]
        if sid == "CLM" and len(els) > 2:
            summary["claim_id"]     = els[1]
            summary["total_charge"] = els[2]
        elif sid == "NM1" and len(els) > 4:
            if els[1] == "IL":   # Subscriber
                summary["patient_last"]  = els[3]
                summary["patient_first"] = els[4]
                if len(els) > 9:
                    summary["member_id"] = els[9]
            elif els[1] == "85":  # Billing Provider
                summary["provider_name"] = els[3]
                if len(els) > 9:
                    summary["provider_npi"] = els[9]
            elif els[1] == "PR":  # Payer
                summary["payer_name"] = els[3]
        elif sid == "HI" and len(els) > 1:
            summary["icd_code"] = els[1].replace("ABK:", "")
        elif sid == "SV1" and len(els) > 2:
            summary["cpt_code"] = els[1].replace("HC:", "")
            summary["charge"]   = els[2]
        elif sid == "DTP" and len(els) > 3 and els[1] == "472":
            raw = els[3]
            if len(raw) == 8:
                summary["service_date"] = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        elif sid == "DMG" and len(els) > 3:
            raw = els[2]
            if len(raw) == 8:
                summary["dob"] = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
            summary["gender"] = "Male" if els[3] == "M" else "Female"
    return summary
