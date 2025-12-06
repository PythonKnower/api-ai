"""
A tiny pure-Python DSL-style wrapper for working with AI providers.

This module exposes a simple, stateful API object via top-level helper
functions so you can write scripts with a compact syntax like:

    from api_ai import API_PROVIDER, API_KEY, API_MODEL, API

    API_PROVIDER(Claude/OpenAI/Google/xAI)
    API_KEY("API_KEY_HERE")
    API_MODEL("model")
    API.note("Type your note here to let the AI use it..")
    resp = API.process()
    API.print(resp)

Important: this is a pure-Python local shim and does not contact external
services. It provides a consistent interface and simple simulation hooks
that you can later extend to call real provider SDKs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import json
import random


class APIError(Exception):
    pass


@dataclass
class APIClient:
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    think_enabled: bool = False
    notes: List[str] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)
    last_response: Any = None

    def set_provider(self, provider: str):
        self.provider = provider
        return self

    def set_key(self, key: str):
        self.api_key = key
        return self

    def set_model(self, *, model: str):
        self.model = model
        return self

    def fail(self, template: str, e: Optional[Exception] = None, when: str = "after"):
        """Record an error message and raise APIError.

        template can include '{e}' which will be replaced by the exception
        message. The `when` argument is informational ("before", "after").
        """
        emsg = str(e) if e is not None else "<no-exception>"
        message = template.replace("{e}", emsg)
        # attach timing and provider info
        full = f"[API.fail when={when} provider={self.provider} model={self.model}] {message}"
        # store in memory/notes for diagnostics
        self.memory.append(full)
        # raise a typed error so callers can catch
        raise APIError(full)

    def print(self, response: Any):
        """Pretty-print a response and remember it as last_response."""
        self.last_response = response
        # if it's a dict-like structure, pretty print JSON
        try:
            if isinstance(response, (dict, list)):
                print(json.dumps(response, indent=2, ensure_ascii=False))
            else:
                print(response)
        except Exception:
            # fall back to simple repr
            print(repr(response))
        return response

    def think(self, enabled: bool = True):
        """Enable or disable "thinking" mode.

        When enabled, calls to `process` and `generate` will include a small
        simulated delay and a descriptive status in the returned result.
        """
        self.think_enabled = bool(enabled)
        return self.think_enabled

    def note(self, text: str):
        """Add a note to persistent memory. Useful for instructions/prompts."""
        self.notes.append(text)
        return text

    # --- core operations -------------------------------------------------
    def _require_key(self):
        if not self.api_key:
            # simulate provider requiring a key; do not block local-only ops
            raise APIError("API key required for this provider operation")

    def _simulate_latency(self):
        if self.think_enabled:
            # small sleep to emulate thinking; keep it short
            time.sleep(0.05 + random.random() * 0.05)

    def process(self, kind: str, data: Any, **kwargs) -> Dict[str, Any]:
        """Process input data (text/image/video) and return a simulated result.

        This is intentionally generic and does not call external services.
        """
        kind = kind.lower()
        # if provider set and operation 'heavy', require key
        if self.provider and self.provider.lower() != "local":
            # simulate that image/video processing requires a key
            if kind in ("image", "video"):
                self._require_key()

        self._simulate_latency()

        result = {
            "provider": self.provider or "local",
            "model": self.model or "default",
            "kind": kind,
            "input_summary": self._summarize_input(data),
            "notes": list(self.notes),
            "timestamp": time.time(),
        }

        # when processing text, return a simple transformation
        if kind == "text":
            text = str(data)
            # if model hints at summarizer, produce shorter text
            if self.model and "summ" in (self.model or "").lower():
                out = self._shorten_text(text)
            else:
                out = text.strip() + "\n\n[processed]"
            result["output"] = out
        else:
            # for image/video return a stub description
            result["output"] = f"<simulated {kind} processing result with size={len(str(data))}>"

        self.last_response = result
        self.memory.append(f"processed:{kind}")
        return result

    def generate(self, kind: str, prompt: Any, **kwargs) -> Dict[str, Any]:
        """Generate content (text/image/video) based on a prompt.

        This is a simulated generator. For real usage, replace body with
        provider SDK calls.
        """
        kind = kind.lower()
        if self.provider and self.provider.lower() != "local":
            # require API key for generation
            self._require_key()

        self._simulate_latency()

        if kind == "text":
            out = self._generate_text_from_prompt(prompt)
        elif kind == "image":
            out = f"<simulated-image size=512x512 from prompt='{prompt}'>"
        elif kind == "video":
            out = f"<simulated-video length=3s from prompt='{prompt}'>"
        else:
            out = f"<unknown-kind:{kind}>"

        response = {
            "provider": self.provider or "local",
            "model": self.model or "default",
            "kind": kind,
            "prompt": str(prompt),
            "output": out,
            "notes": list(self.notes),
            "timestamp": time.time(),
        }
        self.last_response = response
        self.memory.append(f"generated:{kind}")
        return response

    # --- small helpers --------------------------------------------------
    def _summarize_input(self, data: Any) -> str:
        s = repr(data)
        if len(s) > 120:
            return s[:120] + "..."
        return s

    def _shorten_text(self, text: str) -> str:
        # naive shorten: return first line or first 80 chars
        lines = text.strip().splitlines()
        if lines:
            return (lines[0][:200] + ("..." if len(lines[0]) > 200 else ""))
        return text[:200]

    def _generate_text_from_prompt(self, prompt: Any) -> str:
        p = str(prompt).strip()
        # echo plus a small random expansion to feel "creative"
        suffixes = ["\n\nAssistant:", "\n\nGenerated:", "\n\nResult:"]
        chosen = random.choice(suffixes)
        body = p
        # simple paraphrase: reverse a few words and append
        words = p.split()
        if len(words) > 6:
            snippet = " ".join(words[:6]) + "..."
        else:
            snippet = p
        return f"{snippet}{chosen} {snippet[::-1]}"


# singleton default client and convenience functions --------------------
_DEFAULT = APIClient()


def API_PROVIDER(provider: str):
    return _DEFAULT.set_provider(provider)


def API_KEY(key: str):
    return _DEFAULT.set_key(key)


def API_MODEL(*, model: str):
    return _DEFAULT.set_model(model=model)


# expose methods via a simple object for dot-syntax like the user asked
class _APIAlias:
    def fail(self, template: str, e: Optional[Exception] = None, when: str = "after"):
        return _DEFAULT.fail(template, e=e, when=when)

    def print(self, response: Any):
        return _DEFAULT.print(response)

    def process(self, kind: str, data: Any, **kwargs):
        return _DEFAULT.process(kind, data, **kwargs)

    def generate(self, kind: str, prompt: Any, **kwargs):
        return _DEFAULT.generate(kind, prompt, **kwargs)

    def think(self, enabled: bool = True):
        return _DEFAULT.think(enabled=enabled)

    def note(self, text: str):
        return _DEFAULT.note(text)


# instance named API for user convenience
API = _APIAlias()


# exports
__all__ = [
    "API",
    "API_PROVIDER",
    "API_KEY",
    "API_MODEL",
    "APIError",
]
