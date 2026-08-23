"""memory_wire — 런타임 메모리 객체의 **타입-태그 직렬화**.

서버(웹)와 커넥터(로컬)가 같은 계정 메모리 vault 를 공유하려면, 로컬 실행이
메모리 연산을 서버로 RPC 해야 한다(vault 는 서버 소유 — 무발산의 핵심). 그
RPC 의 인자/결과는 런타임 메모리 dataclass/enum(Note, NoteDraft, Turn, Scope …)
이다.

양쪽(서버·사이드카)에 **런타임이 있으므로**, 여기서 타입-태그 방식으로
직렬화하면 어느 쪽이든 원래 타입으로 정확히 복원한다 — MemoryProvider
프로토콜의 35+ 메서드를 일일이 나열하지 않고 remote_memory.py 가 반사 프록시로
덮을 수 있다.

와이어 규약(예약 키 — 평범한 dict 엔 없다고 가정):
  dataclass  → {"$t": "Note", "f": {field: <enc>, ...}}
  enum       → {"$e": "Scope", "v": "session"}
  datetime   → {"$dt": "<iso8601>"}
  date       → {"$d": "<iso>"}
  tuple      → {"$tup": [<enc>, ...]}
  set        → {"$set": [<enc>, ...]}
  bytes      → {"$b": "<base64>"}
  list       → [<enc>, ...]
  dict       → {k: <enc>, ...}
  원시        → 그대로(None/bool/int/float/str)
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as _dt
import enum
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger("xgen_agent_runtime.host.memory_wire")

_REG: Optional[Dict[str, type]] = None


def _registry() -> Dict[str, type]:
    """런타임 메모리 모듈의 공개 dataclass/enum 이름→타입 맵.

    복원은 이 맵으로 타입을 찾는다(태그가 이름을 실어 옴). 미등록 이름은
    복원 시 dict/원시로 격하(진단 가능, 크래시 없음)."""
    reg: Dict[str, type] = {}

    def _scan(module_name: str) -> None:
        try:
            import importlib

            mod = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 — 런타임 부재(비-사이드카 문맥)면 건너뛴다.
            return
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if not isinstance(obj, type):
                continue
            if dataclasses.is_dataclass(obj) or issubclass(obj, enum.Enum):
                # 먼저 등록된 정식 타입을 재-export 별칭이 덮지 않게 한다.
                reg.setdefault(name, obj)

    # 메모리 패키지가 1차 소스. 다만 RPC 결과에는 **다른 모듈**의 dataclass 도
    # 실려 온다 — 특히 검색 결과 RetrievalResult.chunks 의 MemoryChunk 는
    # stages.s02_context.types 에 산다. 이 타입들을 등록하지 않으면 load 가
    # 원시 dict 로 격하하고, s02 가 chunk.content 를 읽다 'dict' object has no
    # attribute 'content' 로 커넥터 로컬 턴이 통째로 죽는다(2026-08-23 실기).
    for module_name in (
        "xgen_agent_runtime.memory",
        "xgen_agent_runtime.stages.s02_context.types",
    ):
        _scan(module_name)
    return reg


def _reg() -> Dict[str, type]:
    global _REG
    if _REG is None:
        _REG = _registry()
    return _REG


def reset_registry() -> None:
    """테스트 훅 — 등록 캐시 무효화(런타임 모듈 재로드 후)."""
    global _REG
    _REG = None


def dump(obj: Any) -> Any:
    """런타임 메모리 객체 → JSON 직렬화 가능한 태그 형태."""
    # ⚠ enum 을 원시보다 **먼저** 판정한다 — Scope/Importance 는 ``str, Enum``
    # 이라 str 체크에 먼저 걸리면 태그를 잃고 bare 문자열이 된다(왕복 시 enum
    # 이 아니라 str 로 복원 → 동일성 깨짐).
    if isinstance(obj, enum.Enum):
        return {"$e": type(obj).__name__, "v": obj.value}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, _dt.datetime):
        return {"$dt": obj.isoformat()}
    if isinstance(obj, _dt.date):
        return {"$d": obj.isoformat()}
    if isinstance(obj, bytes):
        return {"$b": base64.b64encode(obj).decode("ascii")}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "$t": type(obj).__name__,
            "f": {f.name: dump(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
        }
    if isinstance(obj, dict):
        return {str(k): dump(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dump(x) for x in obj]
    if isinstance(obj, tuple):
        return {"$tup": [dump(x) for x in obj]}
    if isinstance(obj, (set, frozenset)):
        return {"$set": [dump(x) for x in obj]}
    # 알 수 없는 타입 — repr 로 격하(진단용, 왕복 불가하지만 크래시 안 함).
    return {"$repr": repr(obj)}


def load(data: Any) -> Any:
    """dump 의 역 — 태그를 보고 원래 런타임 타입으로 복원."""
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [load(x) for x in data]
    if not isinstance(data, dict):
        return data
    if "$t" in data:
        cls = _reg().get(data["$t"])
        fields = {k: load(v) for k, v in (data.get("f") or {}).items()}
        if cls is None:
            # 미등록 타입 — dict 로 격하. 소비자가 속성 접근을 하면 죽으므로
            # (예: s02 의 chunk.content) 등록 누락을 눈에 띄게 남긴다.
            logger.warning(
                "memory_wire: 미등록 타입 %r 을(를) dict 로 격하 — _registry() 에 "
                "정의 모듈을 추가하세요(속성 접근 소비자가 깨질 수 있음).",
                data["$t"],
            )
            return fields
        return _construct(cls, fields)
    if "$e" in data:
        cls = _reg().get(data["$e"])
        if cls is None:
            return data["v"]
        try:
            return cls(data["v"])
        except Exception:  # noqa: BLE001 — 값이 enum 에 없음 → 원시로.
            return data["v"]
    if "$dt" in data:
        try:
            return _dt.datetime.fromisoformat(data["$dt"])
        except Exception:  # noqa: BLE001
            return data["$dt"]
    if "$d" in data:
        try:
            return _dt.date.fromisoformat(data["$d"])
        except Exception:  # noqa: BLE001
            return data["$d"]
    if "$b" in data:
        try:
            return base64.b64decode(data["$b"])
        except Exception:  # noqa: BLE001
            return b""
    if "$tup" in data:
        return tuple(load(x) for x in data["$tup"])
    if "$set" in data:
        return set(load(x) for x in data["$set"])
    if "$repr" in data:
        return data["$repr"]
    return {k: load(v) for k, v in data.items()}


def _construct(cls: type, fields: Dict[str, Any]) -> Any:
    """dataclass 재구성 — 알려진 필드만 넘겨 생성(스키마 드리프트 관대).

    필수 필드 누락 등으로 생성자가 실패하면 빈 인스턴스 + setattr 로 관대하게
    복원(frozen dataclass 는 object.__setattr__). 이렇게 하면 서버·로컬 런타임
    버전이 살짝 달라도 왕복이 크래시하지 않는다."""
    valid = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in fields.items() if k in valid}
    try:
        return cls(**kwargs)
    except Exception:  # noqa: BLE001
        obj = object.__new__(cls)  # type: ignore[call-overload]
        for k, v in fields.items():
            try:
                object.__setattr__(obj, k, v)
            except Exception:  # noqa: BLE001
                pass
        return obj
