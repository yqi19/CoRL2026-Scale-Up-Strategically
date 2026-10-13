#!/usr/bin/env python3
"""
Minimal stand-alone client for the GR00T zmq PolicyServer
(gr00t/policy/server_client.py :: PolicyServer).

Lives in the ManiSkill venv, which deliberately does NOT have the heavy
`gr00t` package installed.  It re-implements exactly the wire format used by
`gr00t.policy.server_client.MsgSerializer`:

  * msgpack for the envelope
  * numpy arrays serialised with ``np.save`` into ``{"__ndarray_class__": True,
    "as_npy": <bytes>}``

Only the ``get_action`` / ``reset`` / ``ping`` endpoints are used here, none of
which return a ``ModalityConfig``, so we do not need to model that class.
"""
from __future__ import annotations

import io
from typing import Any

import msgpack
import numpy as np
import zmq


def _encode(obj: Any):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class GrootClient:
    """REQ-socket client mirroring gr00t.policy.server_client.PolicyClient."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555,
                 timeout_ms: int = 120_000):
        self._ctx = zmq.Context.instance()
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._connect()

    def _connect(self):
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{self._host}:{self._port}")

    def _call(self, endpoint: str, data: dict | None = None,
              requires_input: bool = True):
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            self._sock.send(msgpack.packb(req, default=_encode))
            msg = self._sock.recv()
        except zmq.error.Again:
            self._sock.close()
            self._connect()
            raise
        resp = msgpack.unpackb(msg, object_hook=_decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._call("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._sock.close()
            self._connect()
            return False

    def get_action(self, observation: dict, options: dict | None = None):
        """Returns (action_dict, info_dict)."""
        resp = self._call("get_action",
                          {"observation": observation, "options": options})
        return tuple(resp)  # msgpack list -> (action, info)

    def reset(self, options: dict | None = None):
        return self._call("reset", {"options": options})

    def kill_server(self):
        try:
            self._call("kill", requires_input=False)
        except Exception:
            pass
