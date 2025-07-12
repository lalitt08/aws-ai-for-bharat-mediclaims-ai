# tools/code_mapper.py

# Dummy mapping for demo purposes
CPT_CODE_FIXES = {
    "XYZ123": "99213",   # invalid → valid
    "123ABC": "99214",
}

ICD_CODE_FIXES = {
    "ICDX01": "E11.9",
    "ABC999": "I10",
}

def correct_codes(codes: list) -> list:
    """
    Corrects CPT/ICD codes using predefined rules.
    Returns a new list with valid codes.
    """
    corrected = []

    for code in codes:
        if code in CPT_CODE_FIXES:
            corrected.append(CPT_CODE_FIXES[code])
        elif code in ICD_CODE_FIXES:
            corrected.append(ICD_CODE_FIXES[code])
        else:
            corrected.append(code)  # leave unchanged if not known

    return corrected
