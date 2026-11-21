"""Static consistency checks for the empirical paper; no TeX installation required."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
target = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "final_paper.tex")
tex = target.read_text(encoding="utf-8")
errors = []

bib = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
    for key in match.group(1).split(","):
        if key.strip() not in bib:
            errors.append(f"missing bibliography key: {key.strip()}")

labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
for cmd in ("ref", "eqref"):
    for key in re.findall(rf"\\{cmd}\{{([^}}]+)\}}", tex):
        if key not in labels:
            errors.append(f"missing label for \\{cmd}: {key}")

for path in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
    if not (ROOT / path).is_file():
        errors.append(f"missing figure: {path}")

if tex.count("{") != tex.count("}"):
    errors.append("unbalanced braces")

environment_stack = []
for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex):
    action, name = match.groups()
    if action == "begin":
        environment_stack.append(name)
    elif not environment_stack or environment_stack.pop() != name:
        errors.append(f"unbalanced or incorrectly nested environment: {name}")
if environment_stack:
    errors.append("unclosed environment(s): " + ", ".join(environment_stack))

if errors:
    print("FAILED")
    print("\n".join(f"- {e}" for e in errors))
    sys.exit(1)
print("PASSED: citations, labels, figures, braces, and environments")
