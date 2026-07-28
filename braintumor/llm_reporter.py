"""braintumor/llm_reporter.py - Multi-LLM Clinical Reporting Module (STANDALONE).

Isolated "brain" of the reporter. It does NOT import or modify the pipeline / GUI,
so it is safe to develop independently.

Design
------
* Three reporters, each an OpenAI-compatible chat endpoint (OpenRouter / Groq):
    - Claude 3.5 Sonnet     (synthesizer + reporter)
    - Meditron-7B           (biomedical)
    - Llama-3-OpenBioLLM-8B (biomedical)
* ``generate_all_reports()`` calls all three CONCURRENTLY (asyncio).
* ``generate_consensus()`` has Sonnet synthesize the three into one opinion.
* "Senior consultant neuroradiologist" persona for every request.
* GRACEFUL DEGRADATION: if an API key is missing or a call fails, that reporter
  returns a clearly-labelled OFFLINE TEMPLATE built from the metrics, so the
  module always produces useful console output (e.g. for a demo) without keys.

Keys (set the ones you use):
    OPENROUTER_API_KEY     # for any provider whose base_url is OpenRouter
    GROQ_API_KEY           # for any provider whose base_url is Groq
Optional model-id overrides: SONNET_MODEL, MEDITRON_MODEL, OPENBIO_MODEL.

IMPORTANT (honesty / safety):
    * Calling these APIs SENDS the metrics to an EXTERNAL service - use synthetic
      or de-identified data only; do not transmit PHI without proper agreements.
    * Meditron / OpenBioLLM are research models, NOT clinically validated.
    * Output is AI-generated assistance, NOT a diagnosis. The disclaimer is kept
      in every report on purpose.
    * Confirm each model id is actually hosted by your chosen provider; not all
      biomedical models are on OpenRouter/Groq. Unavailable ones fall back to the
      offline template automatically.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DISCLAIMER = (
    "AI-generated radiology-style assistance for research/education only. NOT a "
    "medical diagnosis. Findings reflect automated image measurements and language "
    "models (some not clinically validated) and must be confirmed by a qualified "
    "neuroradiologist."
)

SENIOR_RADIOLOGIST_SYSTEM = (
    "You are a senior consultant neuroradiologist writing a structured preliminary "
    "report from automated quantitative MRI measurements. Be precise, cautious and "
    "non-alarmist. Use sections: Summary, Findings, Differential considerations, "
    "Recommended next steps, Limitations. Do NOT assert a definitive diagnosis or "
    "grade - frame everything as AI-assisted findings requiring clinician "
    "confirmation. Use only the numbers provided; never invent values. Keep the "
    "disclaimer intact."
)


# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    name: str                       # display name
    model: str                      # model id on the provider
    base_url: str                   # OpenAI-compatible base (…/v1)
    api_key_env: str                # env var holding the API key
    role: str = "reporter"          # "reporter" | "synthesizer"
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


_OPENROUTER = "https://openrouter.ai/api/v1"
_GROQ = "https://api.groq.com/openai/v1"
_OR_HEADERS = {"HTTP-Referer": "https://mindscan.local", "X-Title": "MindScan"}


def default_providers() -> List[LLMConfig]:
    """Best-effort defaults. Override model ids via env or pass your own list.
    (Verify each id is hosted by the provider; unavailable -> offline fallback.)"""
    return [
        LLMConfig("Claude 3.5 Sonnet",
                  os.environ.get("SONNET_MODEL", "anthropic/claude-3.5-sonnet"),
                  _OPENROUTER, "OPENROUTER_API_KEY", role="synthesizer",
                  extra_headers=_OR_HEADERS),
        LLMConfig("Meditron-7B",
                  os.environ.get("MEDITRON_MODEL", "epfl-llm/meditron-7b"),
                  _OPENROUTER, "OPENROUTER_API_KEY", extra_headers=_OR_HEADERS),
        LLMConfig("Llama-3-OpenBioLLM-8B",
                  os.environ.get("OPENBIO_MODEL", "aaditya/Llama3-OpenBioLLM-8B"),
                  _OPENROUTER, "OPENROUTER_API_KEY", extra_headers=_OR_HEADERS),
    ]


# ---------------------------------------------------------------------------
# Low-level OpenAI-compatible chat call (stdlib only; no extra dependency).
# ---------------------------------------------------------------------------
def _chat_sync(cfg: LLMConfig, messages: List[Dict[str, str]],
               temperature: float = 0.3, max_tokens: int = 800,
               timeout: int = 60) -> str:
    key = cfg.api_key
    if not key:
        raise RuntimeError(f"missing API key in env ${cfg.api_key_env}")
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": cfg.model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               **cfg.extra_headers}
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:200]
        raise RuntimeError(f"HTTP {e.code} from {cfg.name}: {detail}") from None
    return body["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
class MedicalReporter:
    """Generates multi-LLM clinical reports from a metrics dict, then a consensus."""

    def __init__(self, metrics: Dict, providers: Optional[List[LLMConfig]] = None,
                 offline_fallback: bool = True):
        self.metrics = dict(metrics or {})
        self.providers = providers or default_providers()
        self.offline_fallback = offline_fallback

    # ----- prompts -----
    def _metrics_block(self) -> str:
        m = self.metrics
        conf = m.get("confidence")
        conf_s = f"{conf*100:.1f}%" if isinstance(conf, (int, float)) else str(conf)
        return (
            f"- Predicted class: {m.get('prediction', 'unknown')}\n"
            f"- Model confidence: {conf_s}\n"
            f"- Estimated tumor area: {m.get('tumor_area_cm2', 'n/a')} cm^2\n"
            f"- Brain-area coverage: {m.get('brain_coverage_pct', 'n/a')}%\n"
            f"- Tumor location: {m.get('tumor_location', 'n/a')}"
        )

    def _user_prompt(self) -> str:
        return (
            "Write the preliminary report from these automated MRI measurements. "
            "Follow the section structure and the cautious framing.\n\n"
            f"MEASUREMENTS:\n{self._metrics_block()}\n\n"
            f"Always end with this exact disclaimer:\n\"{DISCLAIMER}\""
        )

    def _messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": SENIOR_RADIOLOGIST_SYSTEM},
                {"role": "user", "content": self._user_prompt()}]

    # ----- offline template (used when a provider is unavailable) -----
    def _offline_report(self, cfg: LLMConfig, reason: str) -> str:
        m = self.metrics
        conf = m.get("confidence")
        conf_s = f"{conf*100:.1f}%" if isinstance(conf, (int, float)) else str(conf)
        pred = str(m.get("prediction", "unknown"))
        return (
            f"[OFFLINE TEMPLATE - {cfg.name} not called ({reason}). "
            f"Set ${cfg.api_key_env} for a live LLM report.]\n\n"
            "PRELIMINARY AI-ASSISTED NEURORADIOLOGY REPORT\n"
            "Summary:\n"
            f"  Automated analysis favors {pred} (confidence {conf_s}).\n"
            "Findings:\n"
            f"  - Lesion location: {m.get('tumor_location', 'n/a')}.\n"
            f"  - Estimated area {m.get('tumor_area_cm2', 'n/a')} cm^2, "
            f"covering ~{m.get('brain_coverage_pct', 'n/a')}% of the brain area.\n"
            "Differential considerations:\n"
            f"  - Consistent with {pred}; correlate with contrast-enhanced sequences "
            "and clinical history.\n"
            "Recommended next steps:\n"
            "  - Specialist neuroradiology review; advanced sequences as indicated.\n"
            "Limitations:\n"
            "  - Measurements are from a weak/auto segmentation; not a measured volume.\n\n"
            f"{DISCLAIMER}"
        )

    # ----- async core -----
    async def _one(self, cfg: LLMConfig,
                   messages: Optional[List[Dict[str, str]]] = None) -> str:
        messages = messages or self._messages()
        try:
            return await asyncio.to_thread(_chat_sync, cfg, messages)
        except Exception as exc:
            if self.offline_fallback:
                return self._offline_report(cfg, str(exc))
            return f"[ERROR - {cfg.name}: {exc}]"

    async def generate_all_reports(self) -> Dict[str, str]:
        """Call all three reporters CONCURRENTLY. Returns {model_name: report}."""
        results = await asyncio.gather(*(self._one(c) for c in self.providers))
        return {c.name: r for c, r in zip(self.providers, results)}

    async def generate_consensus(self,
                                 reports: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Have the synthesizer (Sonnet) merge the three reports into one opinion.
        Returns {'reports': {...}, 'consensus': str}."""
        if reports is None:
            reports = await self.generate_all_reports()

        synth = next((c for c in self.providers if c.role == "synthesizer"),
                     self.providers[0])
        joined = "\n\n".join(f"### Report from {name}:\n{text}"
                             for name, text in reports.items())
        messages = [
            {"role": "system", "content": SENIOR_RADIOLOGIST_SYSTEM},
            {"role": "user", "content": (
                "Three AI reports were produced for the SAME case (measurements "
                f"below). As the senior consultant, synthesize them into ONE final "
                "consensus opinion: state where they agree, flag any disagreement, "
                "and give a single cautious recommendation. Keep the disclaimer.\n\n"
                f"MEASUREMENTS:\n{self._metrics_block()}\n\n{joined}\n\n"
                f"End with:\n\"{DISCLAIMER}\"")},
        ]
        try:
            consensus = await asyncio.to_thread(_chat_sync, synth, messages)
        except Exception as exc:
            consensus = self._offline_consensus(reports, str(exc))
        return {"reports": reports, "consensus": consensus}

    def _offline_consensus(self, reports: Dict[str, str], reason: str) -> str:
        names = ", ".join(reports.keys())
        m = self.metrics
        return (
            f"[OFFLINE CONSENSUS - synthesizer not called ({reason}).]\n\n"
            f"Consensus across {len(reports)} models ({names}): all reports were "
            f"generated from the same measurements, favoring "
            f"{m.get('prediction', 'unknown')} at "
            f"{(m.get('confidence') or 0)*100:.1f}% confidence, located "
            f"{m.get('tumor_location', 'n/a')}. Recommendation: specialist "
            "neuroradiology review with contrast-enhanced MRI.\n\n" + DISCLAIMER
        )

    # ----- sync convenience wrappers (for scripts/notebooks) -----
    def run_all(self) -> Dict[str, str]:
        return asyncio.run(self.generate_all_reports())

    def run_consensus(self) -> Dict[str, str]:
        return asyncio.run(self.generate_consensus())

    def status(self) -> Dict[str, str]:
        """Per-provider live/offline status (does NOT call the APIs)."""
        return {c.name: ("LIVE" if c.api_key else f"OFFLINE (set ${c.api_key_env})")
                for c in self.providers}


# ---------------------------------------------------------------------------
# Segmentation-quality insight (for the Streamlit "AI Co-Pilot")
# ---------------------------------------------------------------------------
def _offline_seg_insight(report: Dict, reason: str) -> str:
    neck = report.get("neck_or_skull", {}) or {}
    clean = report.get("no_neck_or_skull", {}) or {}
    gap = report.get("neck_gap_dice")
    thr = report.get("success_dice_threshold", 0.70)
    pruned = ("appears to be pruning the hemisphere-collapse artifact effectively"
              if (neck.get("mean_dice") or 0) >= 0.6 else
              "is still challenged on neck/skull slices")
    return (
        f"[OFFLINE INTERPRETATION - LLM not called ({reason}). Set an API key for a "
        "live report.]\n\n"
        "## Segmentation Quality Interpretation\n"
        "**Summary:** Automated masks were scored against the clinician-verified "
        "ground truth across "
        f"{report.get('total_evaluations', 0)} evaluation(s).\n\n"
        "**Neck-artifact robustness:** On slices flagged with neck/lower-skull tissue "
        f"(n={neck.get('n', 0)}), the Compact-Blob localizer reached a mean Dice of "
        f"{neck.get('mean_dice')} with a success rate of {neck.get('success_rate')} "
        f"(Dice >= {thr}). The filter {pruned}.\n\n"
        "**Alignment quality:** Clean slices (n="
        f"{clean.get('n', 0)}) reached mean Dice {clean.get('mean_dice')}. "
        + (f"The neck-vs-clean Dice gap is {gap}. " if gap is not None else "")
        + "Higher Dice/IoU indicates tighter agreement with the radiologist's mask.\n\n"
        "**Caveats:** Metrics are image-level and depend on the clinician mask quality; "
        "they are decision support, not a diagnosis.\n\n" + DISCLAIMER
    )


def segmentation_insight(report: Dict, provider: Optional[LLMConfig] = None,
                         offline_fallback: bool = True) -> str:
    """Interpret a segmentation-evaluation report (from the FastAPI
    ``/api/segmentation-report`` endpoint) for the attending radiologist.

    Uses the synthesizer LLM (Claude 3.5 Sonnet by default) via the same
    OpenAI-compatible transport as the reporters; falls back to a deterministic
    offline interpretation when no key/model is available, so the Streamlit
    Co-Pilot always renders.
    """
    cfg = provider or next((c for c in default_providers() if c.role == "synthesizer"),
                           default_providers()[0])
    user = (
        "Interpret the following automated brain-MRI SEGMENTATION-QUALITY report "
        "for the attending radiologist. Specifically: (1) confirm whether the "
        "'Compact-Blob' filter is successfully pruning the hemisphere-collapse / "
        "neck-skull leakage (compare neck vs. clean Dice/IoU and the success rate), "
        "and (2) comment on the quantitative alignment quality against the "
        "clinician ground truth. Use sections: Summary, Neck-artifact robustness, "
        "Alignment quality, Recommendation, Caveats. Use ONLY the numbers given.\n\n"
        "REPORT:\n" + json.dumps(report, indent=2, default=str)
        + f"\n\nEnd with this disclaimer:\n\"{DISCLAIMER}\""
    )
    messages = [{"role": "system", "content": SENIOR_RADIOLOGIST_SYSTEM},
                {"role": "user", "content": user}]
    try:
        return _chat_sync(cfg, messages, max_tokens=700)
    except Exception as exc:
        if offline_fallback:
            return _offline_seg_insight(report, str(exc))
        return f"[ERROR - segmentation_insight: {exc}]"
