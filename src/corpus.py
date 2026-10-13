import re

def load_sections(path="data/handbook.txt"):
    """Each blank-line separated block = heading line + body. Key = first word of the heading."""
    text = open(path, encoding="utf-8").read()
    sections = {}
    for block in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) >= 2 and len(lines[0].split()) <= 4:
            sections[lines[0].lower().split()[0]] = " ".join(lines)
    return sections
