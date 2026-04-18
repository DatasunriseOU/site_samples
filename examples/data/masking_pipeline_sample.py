"""Public-safe data masking example for article references."""


def mask_document_sections(tokens: list[str], mask_token: str = "<mask>") -> list[str]:
    masked = []
    for token in tokens:
        if token.startswith("DOC_"):
            masked.append(mask_token)
        else:
            masked.append(token)
    return masked
