# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime switches and low-overhead diagnostics for Jet models.

The model is commonly executed in vLLM worker processes, so the environment is
the most reliable way to enable diagnostics.  ``JET_DEBUG`` is the canonical
switch.  ``JET-DEBUG`` and the former ``JET_DEBUG_TRACE`` spelling remain
accepted for existing launch scripts.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


DEBUG_ENV_VAR = "JET_DEBUG"
DEBUG_ENV_ALIASES = ("JET-DEBUG", "JET_DEBUG_TRACE")
_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off", ""))

__all__ = [
    "DEBUG_ENV_VAR",
    "DEBUG_ENV_ALIASES",
    "debug_print",
    "is_debug_enabled",
    "set_debug",
    "unset_debug",
    "is_benchmark_mode",
    "set_benchmark_mode",
    "unset_benchmark_mode",
]


def _parse_switch(value: str | None) -> bool:
    """Parse common boolean environment values, defaulting unknown values on."""
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in _FALSE_VALUES:
        return False
    if normalized in _TRUE_VALUES:
        return True
    # An explicitly supplied, non-empty value historically enabled tracing.
    return True


def is_debug_enabled() -> bool:
    """Return whether Jet diagnostics are enabled.

    The canonical variable has precedence when multiple spellings are set,
    which makes ``JET_DEBUG=0`` an explicit way to disable inherited aliases.
    """
    for name in (DEBUG_ENV_VAR, *DEBUG_ENV_ALIASES):
        value = os.environ.get(name)
        if value is not None:
            return _parse_switch(value)
    return False


def set_debug(enabled: bool = True) -> None:
    """Enable or disable diagnostics for the current process and its children."""
    os.environ[DEBUG_ENV_VAR] = "1" if enabled else "0"


def unset_debug() -> None:
    """Disable diagnostics without removing any legacy environment variables."""
    set_debug(False)


def debug_print(
    message: str | Callable[[], Any],
    *,
    condition: bool = True,
) -> None:
    """Print one diagnostic line when enabled.

    ``message`` may be a callable so expensive tensor statistics are not
    calculated while debugging is disabled.  Callers should omit the prefix;
    this function keeps output consistently grep-able across model components.
    """
    if not condition or not is_debug_enabled():
        return
    value = message() if callable(message) else message
    print(f"JET_DEBUG {value}", flush=True)


def is_benchmark_mode() -> bool:
    return _parse_switch(os.environ.get("JETAI_BENCHMARK_MODE"))


def set_benchmark_mode() -> None:
    os.environ["JETAI_BENCHMARK_MODE"] = "1"


def unset_benchmark_mode() -> None:
    os.environ["JETAI_BENCHMARK_MODE"] = "0"
