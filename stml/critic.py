"""The Socratic critic.

For any claim about a world it returns a structured critique:
  - status: target / weak / premise / trivial / contradicted / undetermined
  - support: indices of a minimal set of premises behind the verdict
  - proof: named-rule derivation (for true claims, or for the negation of
    contradicted ones)
  - countermodel: a small world where every premise holds but the claim
    fails (for undetermined claims)
  - question: a Socratic prompt that points at the flaw without solving it
  - explanation: the full answer, revealed only on request

The same object serves people (rendered in the game) and learning agents
(as structured feedback instead of a scalar reward).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from .logic import Claim, Form, Verdict, minimal_support
from .proof import RULES, derive, proof_steps, rules_used
from .world import INDIVIDUAL_NAMES, World


class Status(str, Enum):
    TARGET = "target"              # correct, new, and as strong as possible
    WEAK = "weak"                  # correct, but a stronger claim follows
    PREMISE = "premise"            # already given (maybe reversed)
    TRIVIAL = "trivial"            # All A are A
    CONTRADICTED = "contradicted"  # must be false
    UNDETERMINED = "undetermined"  # does not follow

    @property
    def correct(self) -> bool:
        return self in (Status.TARGET, Status.WEAK, Status.PREMISE, Status.TRIVIAL)


@dataclass
class Critique:
    claim: str                     # claim key
    text: str
    status: Status
    verdict: Verdict
    support: list[int] = field(default_factory=list)
    proof: list[dict] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    countermodel: list[dict] = field(default_factory=list)
    witness: int | None = None     # index into countermodel of the falsifying individual
    question: str = ""
    explanation: str = ""
    stronger: str | None = None    # for weak claims: a stronger true claim (key)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["verdict"] = self.verdict.value
        return d


def _kinds_phrase(kinds: list[str]) -> str:
    if not kinds:
        return "nothing in particular"
    if len(kinds) == 1:
        return f"one of the {kinds[0]}"
    return "one of the " + ", ".join(kinds[:-1]) + f" and one of the {kinds[-1]}"


def _singular(name: str) -> str:
    return name[:-2] if name.endswith("xes") else name[:-1] if name.endswith("s") else name


class SocraticCritic:
    def __init__(self, world: World):
        self.world = world
        self.sem = world.semantics
        self.names = world.names
        self.closure = derive(world.premises, world.n)
        self._targets = set(world.targets)

    def premise_label(self, i: int) -> str:
        return f"P{i + 1}"

    def _labels(self, idx: list[int]) -> str:
        labels = [self.premise_label(i) for i in idx]
        if not labels:
            return "the fact that every kind has members"
        if len(labels) == 1:
            return labels[0]
        return ", ".join(labels[:-1]) + " and " + labels[-1]

    def _proof(self, claim: Claim) -> list[dict]:
        steps = proof_steps(claim, self.closure)
        index = {s.claim: i for i, s in enumerate(steps)}
        out = []
        for s in steps:
            premise_idx = None
            if s.rule == "premise":
                premise_idx = self.world.premises.index(s.claim)
            out.append({
                "claim": s.claim.key(),
                "text": s.claim.text(self.names),
                "rule": s.rule,
                "rule_text": RULES[s.rule],
                "from": [index[p] for p in s.parents],
                "premise": premise_idx,
            })
        return out

    def _countermodel(self, claim: Claim) -> tuple[list[dict], int | None]:
        model = self.sem.find_model([*self.world.premises, claim.negation()])
        assert model is not None
        people = []
        for i, region in enumerate(model):
            kinds = [k for k in range(self.world.n) if region & (1 << k)]
            people.append({"name": INDIVIDUAL_NAMES[i % len(INDIVIDUAL_NAMES)], "kinds": kinds})
        witness = None
        if claim.is_universal:  # the negation is existential: someone breaks the claim
            rs = self.sem.region_set(claim)
            for i, region in enumerate(model):
                if (rs >> region) & 1:
                    witness = i
                    break
        return people, witness

    def critique(self, claim: Claim) -> Critique:
        w, names = self.world, self.names
        text = claim.text(names)
        A, B = names[claim.a], names[claim.b]
        verdict = self.sem.evaluate(w.premises, claim)
        c = Critique(claim=claim.key(), text=text, status=Status.UNDETERMINED, verdict=verdict)

        if verdict == Verdict.ENTAILED:
            c.support = minimal_support(self.sem, w.premises, claim)
            c.proof = self._proof(claim)
            c.rules = rules_used(proof_steps(claim, self.closure))
            canon = claim.canonical()
            if claim.is_trivial():
                c.status = Status.TRIVIAL
                c.question = f"True, but it tells us nothing new: everything that is one of the {A} is one of the {A}."
            elif canon in w.premise_set:
                c.status = Status.PREMISE
                i = next(i for i, p in enumerate(w.premises) if p.canonical() == canon)
                c.support = [i]
                if w.premises[i] != claim:
                    c.question = (f"That is {self.premise_label(i)} said the other way round. "
                                  f"Does order matter for '{claim.form.value.replace('_', ' ')}' statements?")
                else:
                    c.question = f"That is already given ({self.premise_label(i)})."
            elif w.is_weak(claim):
                c.status = Status.WEAK
                if claim.form == Form.SOME:
                    stronger = Claim(Form.ALL, claim.a, claim.b)
                    if not self.sem.entails(w.premises, stronger):
                        stronger = Claim(Form.ALL, claim.b, claim.a)
                else:
                    stronger = Claim(Form.NO, claim.a, claim.b).canonical()
                c.stronger = stronger.key()
                c.question = f"True. But can you say something stronger about the {A} and the {B}?"
                c.explanation = f"Stronger and also true: {stronger.text(names)}."
            else:
                c.status = Status.TARGET
                c.question = f"Correct. It follows from {self._labels(c.support)}."
            if not c.explanation:
                c.explanation = f"It follows from {self._labels(c.support)}; see the proof."

        elif verdict == Verdict.CONTRADICTED:
            c.status = Status.CONTRADICTED
            neg = claim.negation()
            c.support = minimal_support(self.sem, w.premises, neg)
            c.proof = self._proof(neg)
            c.rules = rules_used(proof_steps(neg, self.closure))
            if len(c.support) == 1:
                c.question = (f"This cannot be true. Look at {self._labels(c.support)}: "
                              f"what does it force about the {A} and the {B}?")
                c.explanation = f"It forces the opposite: {neg.text(names)}."
            elif c.support:
                c.question = (f"This cannot be true. Look at {self._labels(c.support)} together: "
                              f"what do they force about the {A} and the {B}?")
                c.explanation = f"They force the opposite: {neg.text(names)}."
            else:
                c.question = "This cannot be true in any world. Why not?"
                c.explanation = f"It is false by logic alone: {neg.text(names)}."

        else:
            c.status = Status.UNDETERMINED
            c.countermodel, c.witness = self._countermodel(claim)
            if claim.is_universal:
                c.question = ("Could every premise be true while your claim is false? "
                              "Try to imagine a single individual that would break it.")
                who = c.countermodel[c.witness]
                kinds = [names[k] for k in who["kinds"]]
                if claim.form == Form.ALL:
                    c.explanation = (f"In this world every premise holds, yet {who['name']} is "
                                     f"{_kinds_phrase(kinds)} and not one of the {B}. "
                                     f"So the premises do not force '{text}'.")
                else:
                    c.explanation = (f"In this world every premise holds, yet {who['name']} is "
                                     f"{_kinds_phrase(kinds)}: both a {_singular(A)} and a {_singular(B)}. "
                                     f"So the premises do not force '{text}'.")
            else:
                c.question = ("Do the premises guarantee that such an individual exists, "
                              "or only that one could?")
                if claim.form == Form.SOME:
                    c.explanation = (f"In this world every premise holds, yet nobody is both "
                                     f"a {_singular(A)} and a {_singular(B)}. So '{text}' does not follow.")
                else:
                    c.explanation = (f"In this world every premise holds, yet every one of the {A} "
                                     f"is also one of the {B}. So '{text}' does not follow.")
        return c

    def critique_all(self) -> dict[str, Critique]:
        from .logic import all_claims
        return {cl.key(): self.critique(cl) for cl in all_claims(self.world.n, self.world.claim_forms)}
