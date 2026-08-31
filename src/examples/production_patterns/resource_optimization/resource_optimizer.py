"""Resource-Aware Optimization example using AWS Bedrock and LangChain.

Demonstrates the **Resource-Aware Optimization** production pattern: the
agent is given an explicit, finite budget (a stand-in for $ / token spend)
and must decide, section by section, how to spend it — not just *whether* to
do a piece of work, but *how expensively* to do it. Each unit of work has
several **tiers** of decreasing cost and quality; the agent walks down that
ladder as the remaining budget shrinks, and it protects required work by
funding it before optional work.

Tier ladder, tried richest-first, for every section:

    PREMIUM   -> full LLM call, thorough multi-sentence analysis (priciest)
    STANDARD  -> LLM call, one terse sentence (cheaper prompt/output)
    ECONOMY   -> no LLM call at all: a cached canned fact (cheapest non-zero)
    PLACEHOLDER -> free static note ("budget exhausted") — required sections
                   only, so a required section is never silently dropped
    SKIP      -> section omitted entirely — optional sections only

Flow for one report:

    for each section, required sections first:
        tier = richest tier whose cost <= remaining budget
                 (PLACEHOLDER if required and nothing fits,
                  SKIP if optional and nothing fits)
        run section at that tier, deduct its cost
    if enough budget remains: LLM synthesises the sections into one brief
    else: sections are joined verbatim, no further LLM spend

Simplifications that keep the example offline and deterministic:
  - There is one shared model (see ``src/app/bedrock.py``); "cheaper" tiers
    are not a smaller model but a cheaper *prompt* (a one-sentence ask
    instead of a thorough one) or no model call at all, which is the same
    cost lever available to a single-model deployment.
  - Costs are fixed numbers declared per section/tier, not metered spend —
    the same stand-in role ``exception_handling``'s scripted failures and
    ``a2a``'s hard-coded rate tables play elsewhere in this repo.
  - The economy tier's "cached fact" is a small hard-coded table, not a real
    cache.

How this differs from the neighbouring patterns in this repo:
  - exception_handling: also walks a ladder per unit of work, but the ladder
    is triggered by a *failure* (retry -> fallback -> degrade). Here every
    step succeeds; the ladder is triggered by *remaining budget*, decided
    before the step ever runs.
  - hitl: a fixed policy intercepts side-effecting calls for a *human* to
    approve. Here the policy is automatic and decides *cost tier*, not
    approve/reject.
  - goal_monitoring: re-attempts the *whole* work product against a
    checklist until it passes or a retry cap is hit. Here the budget is
    spent once, in one pass, across independent sections — there is no
    retry, only tier selection.
  - a2a: dispatches a whole request to one remote agent. Here one report is
    built from several sections, each costed and tiered independently.
"""

import logging
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# Shared ChatBedrock instance, constructed once at import time in
# src/app/bedrock.py (which also validates AWS credentials at module scope).
from ....app.bedrock import llm

logger = logging.getLogger(__name__)

# --- Tier vocabulary ---------------------------------------------------
# Plain string constants (the repo avoids Enum). Tried in this order for
# every section; PLACEHOLDER and SKIP both cost nothing to select but differ
# in what runs: a static note for required work, nothing at all for optional.
PREMIUM: str = "premium"
STANDARD: str = "standard"
ECONOMY: str = "economy"
PLACEHOLDER: str = "placeholder"
SKIP: str = "skip"

_PAID_TIERS: tuple[str, ...] = (PREMIUM, STANDARD, ECONOMY)

# Verbosity instruction injected into the section prompt for each paid,
# LLM-backed tier. This — not a different model — is the cost lever: a
# shorter ask costs less to run than a thorough one.
_TIER_DETAIL: dict[str, str] = {
    PREMIUM: (
        "Write a thorough 3-4 sentence analysis with plausible, specific "
        "detail and one actionable takeaway."
    ),
    STANDARD: "Write exactly one concise, information-dense sentence.",
}

# Fixed cost of the final LLM synthesis pass, charged only if it runs.
SYNTHESIS_COST: float = 3.0


@dataclass
class Budget:
    """A finite resource budget, spent down across a report's sections."""

    total: float
    spent: float = 0.0

    @property
    def remaining(self) -> float:
        return self.total - self.spent

    def spend(self, amount: float) -> None:
        self.spent += amount


@dataclass
class Subtask:
    """One report section: what it costs at each paid tier, and its prompt."""

    name: str
    label: str
    required: bool
    costs: dict[str, float]
    system_prompt: str
    placeholder: str


# --- Report sections --------------------------------------------------------
# Listed required-first so `run_report` funds must-have sections before
# nice-to-have ones when the budget is tight.

SUBTASKS: list[Subtask] = [
    Subtask(
        name="company_overview",
        label="Company Overview",
        required=True,
        costs={PREMIUM: 14.0, STANDARD: 6.0, ECONOMY: 2.0},
        system_prompt=(
            "You are drafting the 'Company Overview' section of a vendor "
            "briefing. {detail} Do not invent precise financials; keep "
            "claims plausible and generic where specifics aren't given."
        ),
        placeholder=(
            "Company Overview: budget exhausted before this section could "
            "be researched — using no data; verify manually before use."
        ),
    ),
    Subtask(
        name="financial_snapshot",
        label="Financial Snapshot",
        required=True,
        costs={PREMIUM: 12.0, STANDARD: 5.0, ECONOMY: 2.0},
        system_prompt=(
            "You are drafting the 'Financial Snapshot' section of a vendor "
            "briefing. {detail} Do not invent precise financials; keep "
            "claims plausible and generic where specifics aren't given."
        ),
        placeholder=(
            "Financial Snapshot: budget exhausted before this section "
            "could be researched — using last-known-filing placeholder; "
            "verify manually before use."
        ),
    ),
    Subtask(
        name="competitor_scan",
        label="Competitor Scan",
        required=False,
        costs={PREMIUM: 16.0, STANDARD: 7.0, ECONOMY: 3.0},
        system_prompt=(
            "You are drafting the 'Competitor Scan' section of a vendor "
            "briefing. {detail} Do not invent precise financials; keep "
            "claims plausible and generic where specifics aren't given."
        ),
        placeholder="Competitor Scan: budget exhausted; section omitted.",
    ),
    Subtask(
        name="risk_flags",
        label="Risk Flags",
        required=False,
        costs={PREMIUM: 9.0, STANDARD: 4.0, ECONOMY: 2.0},
        system_prompt=(
            "You are drafting the 'Risk Flags' section of a vendor "
            "briefing. {detail} Do not invent precise financials; keep "
            "claims plausible and generic where specifics aren't given."
        ),
        placeholder="Risk Flags: budget exhausted; section omitted.",
    ),
]

# Economy tier: a small cached-fact table keyed by company, standing in for
# a real cache lookup (deliberately no LLM call, so it's free of prompt cost
# and fully deterministic — same offline-data approach as `a2a`'s rate table).
_ECONOMY_CACHE: dict[str, dict[str, str]] = {
    "Northwind Robotics": {
        "company_overview": (
            "Northwind Robotics: mid-size industrial-automation vendor, "
            "HQ Melbourne, ~450 staff (cached profile)."
        ),
        "financial_snapshot": (
            "Northwind Robotics: privately held, last disclosed revenue "
            "band A$80-120M (cached filing summary)."
        ),
        "competitor_scan": (
            "Northwind Robotics: competes with two other regional "
            "automation integrators (cached market note)."
        ),
        "risk_flags": (
            "Northwind Robotics: no material risk flags on file "
            "(cached compliance note)."
        ),
    },
}


def _economy_fact(subtask: Subtask, company: str) -> str:
    cached = _ECONOMY_CACHE.get(company, {}).get(subtask.name)
    return cached or f"{subtask.label}: no cached profile for {company}."


def _subtask_chain(system_template: str) -> Runnable[dict, str]:
    """Stateless prompt|llm|parser chain for one section at a paid tier."""
    prompt = ChatPromptTemplate.from_messages(
        [(("system", system_template)), ("human", "Company: {company}")]
    )
    return prompt | llm | StrOutputParser()


def build_synthesis_chain() -> Runnable[dict, str]:
    """Merge the collected sections into one coherent brief."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are compiling a short vendor briefing from section "
                "drafts of mixed detail. Merge them into a coherent brief "
                "with one paragraph per section, in the order given. Do "
                "not add a section that is not present in the input.",
            ),
            ("human", "Company: {company}\n\nSections:\n{sections}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def select_tier(costs: dict[str, float], remaining: float, required: bool) -> str:
    """Richest tier whose cost fits the remaining budget.

    Falls back to PLACEHOLDER for a required section (it must produce
    something) or SKIP for an optional one (it is dropped instead).
    """
    for tier in _PAID_TIERS:
        if costs[tier] <= remaining:
            return tier
    return PLACEHOLDER if required else SKIP


def run_subtask(subtask: Subtask, tier: str, company: str) -> str:
    """Produce one section's text at the tier already chosen for it."""
    if tier in (PREMIUM, STANDARD):
        chain = _subtask_chain(subtask.system_prompt.format(detail=_TIER_DETAIL[tier]))
        return chain.invoke({"company": company}).strip()
    if tier == ECONOMY:
        return _economy_fact(subtask, company)
    return subtask.placeholder  # PLACEHOLDER


def _format_sections(sections: list[tuple[Subtask, str, str]]) -> str:
    return "\n\n".join(
        f"[{subtask.label} - {tier}]\n{text}" for subtask, tier, text in sections
    )


def run_report(company: str, budget_total: float) -> str:
    """Build one vendor briefing for `company`, spending at most `budget_total`.

    Loop state is just the running `Budget` and the list of section results;
    tier selection for each section only looks at what remains right now.
    """
    budget = Budget(total=budget_total)
    sections: list[tuple[Subtask, str, str]] = []
    print(f"  budget: {budget_total:g} units")

    for subtask in sorted(SUBTASKS, key=lambda s: not s.required):
        tier = select_tier(subtask.costs, budget.remaining, subtask.required)
        if tier == SKIP:
            print(
                f"  [{subtask.name}] optional, remaining {budget.remaining:g} "
                f"< economy cost {subtask.costs[ECONOMY]:g} -> skip"
            )
            continue

        cost = subtask.costs[tier] if tier in _PAID_TIERS else 0.0
        text = run_subtask(subtask, tier, company)
        budget.spend(cost)
        print(
            f"  [{subtask.name}] tier={tier} cost={cost:g} "
            f"remaining={budget.remaining:g}"
        )
        sections.append((subtask, tier, text))

    if budget.remaining >= SYNTHESIS_COST:
        budget.spend(SYNTHESIS_COST)
        print(
            f"  [synthesis] tier={PREMIUM} cost={SYNTHESIS_COST:g} "
            f"remaining={budget.remaining:g}"
        )
        return build_synthesis_chain().invoke(
            {"company": company, "sections": _format_sections(sections)}
        ).strip()

    print(
        f"  [synthesis] remaining {budget.remaining:g} < cost "
        f"{SYNTHESIS_COST:g} -> degrade to plain concatenation"
    )
    return _format_sections(sections)


# Each run holds the task and company fixed and varies only the budget, so
# the same report demonstrates the full tier ladder as spending pressure
# changes — one run per branch, like `exception_handling`'s sample queries.
SAMPLE_RUNS: list[tuple[str, float]] = [
    # Ample budget: every section affords PREMIUM, and enough is left over
    # for a real LLM synthesis pass.
    ("Northwind Robotics", 70.0),
    # Constrained: required sections drop to STANDARD/ECONOMY, one optional
    # section is skipped outright, and synthesis degrades to concatenation.
    ("Northwind Robotics", 10.0),
    # Minimal: required sections run at ECONOMY or fall back to a
    # PLACEHOLDER note, both optional sections are skipped, synthesis
    # degrades.
    ("Northwind Robotics", 3.0),
]


def handle_requests() -> None:
    """Main entry point: run the resource-aware briefing agent over sample budgets.

    Registered in the CLI (see src/examples/__main__.py
    `_run_resource_optimization` and src/app/cli.py
    `CMD_RESOURCE_OPTIMIZATION`), invoked by:
        uv run python -m src.examples resource_optimization
    """
    logger.info(
        "Running resource-aware optimization example over %d sample run(s)",
        len(SAMPLE_RUNS),
    )
    print("Running resource-aware vendor-briefing agent across sample budgets")

    for company, budget_total in SAMPLE_RUNS:
        print("=" * 60)
        print(f"Company: {company} | Budget: {budget_total:g} units")
        report = run_report(company, budget_total)
        print("-" * 60)
        print(report)
        print("=" * 60)
        print()
