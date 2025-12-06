"""
A tiny pure-Python DSL-style wrapper for working with AI providers.

This module exposes a simple, stateful API object via top-level helper
functions so you can write scripts with a compact syntax like:

    from api_ai import API_PROVIDER, API_KEY, API_MODEL, API, API_TEMP

    API_PROVIDER(Claude/OpenAI/Google/xAI)
    API_KEY("API_KEY_HERE")
    API_MODEL("model")
    API.note("Type your note here to let the AI use it..")
    resp = API.process() 
    API.print(resp)
    API.temp()
    API.set_temp

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
    temperature_value: float = 1.0  # default 1.0

    def set_provider(self, provider: str):
        self.provider = provider
        return self

    def set_key(self, key: str):
        self.api_key = key
        return self

    def set_model(self, *, model: str):
        self.model = model
        return self

    def set_temp(self, temp: float):
        """Set generation temperature. Must be between 0.1 and 2."""
        if not (0.1 <= temp <= 2):
            raise APIError("Temperature must be >=0.1 and <=2")
        self.temperature_value = temp
        return self.temperature_value

    def temp(self):
        """Get current temperature value."""
        return self.temperature_value

    def fail(self, template: str, e: Optional[Exception] = None, when: str = "after"):
        emsg = str(e) if e is not None else "<no-exception>"
        message = template.replace("{e}", emsg)
        full = f"[API.fail when={when} provider={self.provider} model={self.model}] {message}"
        self.memory.append(full)
        raise APIError(full)

    def print(self, response: Any):
        self.last_response = response
        try:
            if isinstance(response, (dict, list)):
                print(json.dumps(response, indent=2, ensure_ascii=False))
            else:
                print(response)
        except Exception:
            print(repr(response))
        return response

    def think(self, enabled: bool = True):
        self.think_enabled = bool(enabled)
        return self.think_enabled

    def note(self, text: str):
        self.notes.append(text)
        return text

    # --- core operations -------------------------------------------------
    def _require_key(self):
        if not self.api_key:
            raise APIError("API key required for this provider operation")

    def _simulate_latency(self):
        if self.think_enabled:
            time.sleep(0.05 + random.random() * 0.05)

    def process(self, kind: str, data: Any, **kwargs) -> Dict[str, Any]:
        kind = kind.lower()
        if self.provider and self.provider.lower() != "local":
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

        if kind == "text":
            text = str(data)
            if self.model and "summ" in (self.model or "").lower():
                out = self._shorten_text(text)
            else:
                out = text.strip() + "\n\n[processed]"
            result["output"] = out
        else:
            result["output"] = f"<simulated {kind} processing result with size={len(str(data))}>"

        self.last_response = result
        self.memory.append(f"processed:{kind}")
        return result

    def generate(self, kind: str, prompt: Any, **kwargs) -> Dict[str, Any]:
        kind = kind.lower()
        if self.provider and self.provider.lower() != "local":
            self._require_key()

        self._simulate_latency()

        if kind == "text":
            out = self._generate_text_from_prompt(prompt)
            # optional: apply temperature to simulate creativity/randomness
            noise = int(self.temperature_value * 10)
            out += f"\n\n[temperature={self.temperature_value}, noise={noise}]"
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
            "temperature": self.temperature_value,
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
        lines = text.strip().splitlines()
        if lines:
            return (lines[0][:200] + ("..." if len(lines[0]) > 200 else ""))
        return text[:200]

    def _generate_text_from_prompt(self, prompt: Any) -> str:
        p = str(prompt).strip()
        suffixes = ["\n\nAssistant:", "\n\nGenerated:", "\n\nResult:"]
        chosen = random.choice(suffixes)
        words = p.split()
        snippet = " ".join(words[:6]) + "..." if len(words) > 6 else p
        return f"{snippet}{chosen} {snippet[::-1]}"


# singleton default client
_DEFAULT = APIClient()


def API_PROVIDER(provider: str):
    return _DEFAULT.set_provider(provider)


def API_KEY(key: str):
    return _DEFAULT.set_key(key)


def API_MODEL(*, model: str):
    return _DEFAULT.set_model(model=model)


# alias object for convenient dot syntax
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

    def set_temp(self, temp: float):
        return _DEFAULT.set_temp(temp)

    def temp(self):
        return _DEFAULT.temp()


# instance named API for convenience
API = _APIAlias()


# exports
__all__ = [
    "API",
    "API_PROVIDER",
    "API_KEY",
    "API_MODEL",
    "APIError",
]
