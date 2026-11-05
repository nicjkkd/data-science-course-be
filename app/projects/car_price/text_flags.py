"""Deterministic keyword rules over the listing description (used in train AND serve).

A rule fires when one of its patterns matches inside a clause (text between . , ! ? ; or a
line break) that has no negation word: «без ДТП», «в ДТП не була», «не потребує вкладень» and
«жодного фарбованого елемента» do not fire.
"""

import re

RULES: dict[str, list[str]] = {
    "accident": [r"\bдтп\b", r"після удару", r"\bбит(а|е|ий|і)\b", r"\bудар"],
    "repaint": [r"фарбован", r"перефарбов", r"підфарбов", r"крашен"],
    "recent_repair": [
        r"\bзамін(а|ено|ив|ила|ені|ою)\b",
        r"\bнов(а|ий|е|і) (акб|акумулятор|гума|зчеплення|грм|ланцюг)",
    ],
    "needs_repair": [
        r"потребує (ремонту|вкладень|заміни)",
        r"на запчастини",
        r"не на ходу",
        r"треба (ремонт|робити)",
    ],
    "service_history": [
        r"сервісн(а|у|ою) (книжк|книг|історі)",
        r"історі(я|ю|єю) обслуг",
        r"обслуговув\w* (на|у|в) офіц",
        r"по сервіс",
    ],
    "imported_us": [r"(зі|з|із) сша", r"copart", r"iaai", r"(з|із) америк", r"в америці"],
}
_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in RULES.items()}
# «не на ходу» is itself the needs_repair signal, so it is exempt from the negation check
_NEGATION_EXEMPT = {"не на ходу"}

_CLAUSE = re.compile(r"[.,!?;\n]+")
_WORD = re.compile(r"\w+")
NEGATIONS = {"не", "без", "ні", "нема", "немає", "ніколи", "ніякого", "ніяких"}


def _negated(clause: str, match: re.Match) -> bool:
    if match.group(0).lower() in _NEGATION_EXEMPT:
        return False
    if clause[max(0, match.start() - 2) : match.start()].lower() == "не":  # «непотребує»
        return True
    words = [w.lower() for w in _WORD.findall(clause)]
    return any(w in NEGATIONS or w.startswith(("жодн", "ніяк")) for w in words)


def flags_from_text(text: str | None) -> set[str]:
    if not isinstance(text, str) or not text:  # None / NaN from rows without a description
        return set()
    found: set[str] = set()
    for clause in _CLAUSE.split(text):
        for flag, patterns in _COMPILED.items():
            if flag in found:
                continue
            for p in patterns:
                m = p.search(clause)
                if m and not _negated(clause, m):
                    found.add(flag)
                    break
    return found
