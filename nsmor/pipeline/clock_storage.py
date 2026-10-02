"""Lossless per-trial storage for source clock observations."""
from __future__ import annotations

import json
from typing import Any
import zlib

_ENCODING = "clock-provenance-zlib-json-v1"
_COLUMNS = ("source_row_indices", "raw_sys_time", "raw_ard_time", "time_source")


def pack_clock_provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compress one trial at a time without changing its raw timing tokens."""
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return {"encoding": _ENCODING, "payload": zlib.compress(payload.encode("utf-8"))}


def unpack_clock_provenance(value: Any) -> Any:
    """Restore compact records; leave legacy lists and absent provenance alone."""
    if value is None or isinstance(value, list):
        return value
    if (not isinstance(value, dict) or value.get("encoding") != _ENCODING
            or not isinstance(value.get("payload"), bytes)):
        raise ValueError("Invalid clock provenance encoding")
    try:
        decoder = zlib.decompressobj()
        payload = decoder.decompress(value["payload"]) + decoder.flush()
        if not decoder.eof or decoder.unused_data:
            raise ValueError("Incomplete or trailing clock provenance payload")
        records = json.loads(payload.decode("utf-8"))
    except (zlib.error, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid clock provenance payload") from exc
    if not isinstance(records, list):
        raise ValueError("Clock provenance must contain a list of records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Clock provenance record must be a dictionary")
        columns = [record.get(key) for key in _COLUMNS]
        if (not all(isinstance(column, list) for column in columns)
                or len({len(column) for column in columns}) != 1):
            raise ValueError("Clock provenance source-row columns must align")
        if (any(type(index) is not int or index < 0 for index in columns[0])
                or any(not isinstance(token, str)
                       for column in columns[1:] for token in column)):
            raise ValueError("Clock provenance requires integer rows and exact string tokens")
    return records


def restore_clock_provenance(
    dataset: dict[str, Any], *, materialize: bool = True,
) -> None:
    """Validate each shared trial once; optionally retain its compact storage."""
    decoded: dict[int, Any] = {}
    seen: set[int] = set()

    def restore(value: Any) -> Any:
        if not isinstance(value, dict):
            return unpack_clock_provenance(value)
        key = id(value)
        if not materialize:
            if key not in seen:
                unpack_clock_provenance(value)
                seen.add(key)
            return value
        if key not in decoded:
            decoded[key] = unpack_clock_provenance(value)
        return decoded[key]

    if "source_clock_provenance" in dataset:
        restored = []
        for value in dataset["source_clock_provenance"]:
            value = restore(value)
            if materialize:
                restored.append(value)
        if materialize:
            dataset["source_clock_provenance"] = restored
    for item in dataset.get("labeling_eligibility", []):
        if "clock_provenance" in item:
            value = restore(item["clock_provenance"])
            if materialize:
                item["clock_provenance"] = value
