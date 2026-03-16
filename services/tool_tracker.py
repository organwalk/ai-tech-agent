import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ToolTracker:
    duplicate_limit: int
    round_count: int = 0
    called_tools: List[str] = field(default_factory=list)
    _signature_counter: Counter = field(default_factory=Counter)

    def start_round(self) -> None:
        self.round_count += 1

    def register(self, tool_name: str, tool_args: Dict) -> Tuple[bool, str]:
        signature = f"{tool_name}:{json.dumps(tool_args, ensure_ascii=False, sort_keys=True)}"
        self._signature_counter[signature] += 1
        self.called_tools.append(tool_name)

        if self._signature_counter[signature] > self.duplicate_limit:
            return False, signature
        return True, signature

    def recent_tools(self, limit: int = 5) -> List[str]:
        if limit <= 0:
            return []
        return self.called_tools[-limit:]

