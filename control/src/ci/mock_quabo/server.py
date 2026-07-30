#!/usr/bin/env python3
"""
mock_quabo/server.py

asyncio UDP server simulating PANOSETI quabo hardware.

Scope: models only the packet interface (Quabo-packet-interface.md).
Does NOT simulate firmware bugs or hardware errata — assumes ideal hardware
so test failures are attributable to control-plane code.

Science data: primary path is tcpreplay PCAP injection (existing CI pattern).
The server supports an optional emit_science_packet UDS command for targeted tests.

Command ports (per quabo):
  Q0 → port 60000  (quabo index 0, IP base+0)
  Q1 → port 60001  (quabo index 1, IP base+1)
  Q2 → port 60002  (quabo index 2, IP base+2)
  Q3 → port 60003  (quabo index 3, IP base+3)

HK output: sent every 3 s to the configured hk_dest IP on UDP port 60002.

Control socket: UDS at /tmp/mock_quabo.sock (topology-level commands).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import struct
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("mock_quabo")

# ── Command byte constants ──────────────────────────────────────────────────
CMD_SET_ASICS           = 0x81  # 0x80 | 0x01 — 492-byte packet, echo response
CMD_SET_HVS             = 0x82  # 0x80 | 0x02 — set high voltage
CMD_SET_ACQUISITION     = 0x83  # 0x80 | 0x03 — acquisition mode
CMD_RESET               = 0x84  # 0x80 | 0x04 — reboot quabo
CMD_CHANNEL_MASK        = 0x86  # 0x80 | 0x06 — GOE mask
CMD_CALIBRATE_PH        = 0x07  # calibrate PH baseline → 516-byte response
CMD_SOFTWARE_1PPS       = 0x8F  # 0x80 | 0x0f — software 1PPS
CMD_HK_INTERVAL         = 0x20  # set HK emission interval

HK_PORT = 60002
HK_INTERVAL_SEC = 3.0
HK_PACKET_LEN = 64

CMD_PORTS = [60000, 60001, 60002, 60003]
UDS_SOCK_PATH = os.getenv("MOCK_QUABO_UDS", "/tmp/mock_quabo.sock")

# Module config from env (set in Dockerfile/entrypoint)
MODULE_BASE_IP = os.getenv("MOCK_QUABO_BASE_IP", "0.0.0.0")
MODULE_ID = int(os.getenv("MOCK_QUABO_MODULE_ID", "200"))
HK_DEST_PORT = int(os.getenv("MOCK_QUABO_HK_DEST_PORT", str(HK_PORT)))


@dataclass
class QuaboState:
    """Runtime state for one quabo slot."""
    quabo_index: int
    module_id: int
    uid: str = "DEADBEEF12345678"
    acq_mode: int = 0           # 0 = disabled
    hv_counts: int = 0
    goe_mask: int = 0x3         # any single pixel trigger
    asic_regs: bytes = field(default_factory=lambda: bytes(107))  # 107 bytes per ASIC
    boot_count: int = 0         # increments on CMD_RESET

    @property
    def boardloc(self) -> int:
        return self.module_id * 4 + self.quabo_index


@dataclass
class ServerState:
    """Global server state shared across quabo slot handlers."""
    quabos: list[QuaboState] = field(default_factory=list)
    hk_dest_ip: str = ""
    first_hk_sent: bool = False
    silenced: bool = False      # when True, drop all UDP responses

    def reset(self) -> None:
        for q in self.quabos:
            q.acq_mode = 0
            q.hv_counts = 0
            q.goe_mask = 0x3
            q.asic_regs = bytes(107)
        self.hk_dest_ip = ""
        self.first_hk_sent = False
        self.silenced = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "hk_dest_ip": self.hk_dest_ip,
            "silenced": self.silenced,
            "quabos": [
                {
                    "index": q.quabo_index,
                    "boardloc": q.boardloc,
                    "acq_mode": q.acq_mode,
                    "hv_counts": q.hv_counts,
                    "goe_mask": q.goe_mask,
                    "uid": q.uid,
                    "boot_count": q.boot_count,
                }
                for q in self.quabos
            ],
        }


# ── UDP protocol per quabo slot ─────────────────────────────────────────────

class QuaboProtocol(asyncio.DatagramProtocol):
    """Handles UDP commands for one quabo slot."""

    def __init__(self, state: ServerState, quabo_index: int) -> None:
        self.state = state
        self.quabo_index = quabo_index
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if self.state.silenced:
            return
        if not data:
            return

        cmd = data[0]
        quabo = self.state.quabos[self.quabo_index]
        response: bytes | None = None

        if cmd == CMD_SET_ASICS:
            # 492-byte payload: read back = what was written (ideal hardware)
            quabo.asic_regs = data[1:108] if len(data) >= 108 else data[1:]
            response = data  # echo the full command packet

        elif cmd == CMD_SET_HVS:
            if len(data) >= 3:
                quabo.hv_counts = struct.unpack_from(">H", data, 1)[0]
            response = data

        elif cmd == CMD_SET_ACQUISITION:
            if len(data) >= 2:
                quabo.acq_mode = data[1]
            response = data

        elif cmd == CMD_RESET:
            quabo.boot_count += 1
            quabo.acq_mode = 0
            response = data

        elif cmd == CMD_CHANNEL_MASK:
            if len(data) >= 2:
                quabo.goe_mask = data[1]
            response = data

        elif cmd == CMD_CALIBRATE_PH:
            # Return 516-byte response (command + 512 bytes of baseline data)
            response = data[:4] + bytes(512)

        elif cmd == CMD_SOFTWARE_1PPS:
            response = data

        elif cmd == CMD_HK_INTERVAL:
            if len(data) >= 2:
                pass  # just acknowledge
            response = data

        else:
            # Unknown command: echo if the 0x80 bit is set (request echo convention)
            if cmd & 0x80:
                response = data

        if response and self.transport:
            self.transport.sendto(response, addr)

    def error_received(self, exc: Exception) -> None:
        logger.warning(f"Q{self.quabo_index} UDP error: {exc}")


# ── HK packet emission ───────────────────────────────────────────────────────

def build_hk_packet(quabo: QuaboState, first: bool) -> bytes:
    """Build a 64-byte HK packet per Quabo-packet-interface.md."""
    pkt = bytearray(HK_PACKET_LEN)
    pkt[0] = 0xAA if first else 0x00   # bootbyte
    struct.pack_into(">H", pkt, 2, quabo.boardloc)  # BOARDLOC at offset 2
    # UID: 8 bytes at offset 4
    uid_bytes = bytes.fromhex(quabo.uid)[:8].ljust(8, b"\x00")
    pkt[4:12] = uid_bytes
    # TEMP1 (plausible: 25°C encoded as 250 in some 0.1°C/LSB scale)
    struct.pack_into(">H", pkt, 12, 250)
    # HVMON (plausible idle value)
    struct.pack_into(">H", pkt, 14, 65000)
    # FWVER
    struct.pack_into(">H", pkt, 16, 0x0100)
    return bytes(pkt)


async def hk_emitter(state: ServerState) -> None:
    """Periodically emit HK packets to the configured hk_dest_ip."""
    first = True
    while True:
        await asyncio.sleep(HK_INTERVAL_SEC)
        if not state.hk_dest_ip or state.silenced:
            first = True
            continue
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for quabo in state.quabos:
                pkt = build_hk_packet(quabo, first)
                sock.sendto(pkt, (state.hk_dest_ip, HK_DEST_PORT))
            sock.close()
            first = False
        except OSError as e:
            logger.warning(f"HK emit error: {e}")


# ── UDS control socket ────────────────────────────────────────────────────────

async def handle_control_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    state: ServerState,
) -> None:
    """Handle one UDS control connection.

    Protocol: newline-terminated JSON commands.
    Responses are newline-terminated JSON objects.
    """
    try:
        raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
        line = raw.decode().strip()
        parts = line.split(None, 1)
        cmd = parts[0] if parts else ""
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "set_hk_dest":
            state.hk_dest_ip = arg.strip()
            resp = {"ok": True, "hk_dest_ip": state.hk_dest_ip}

        elif cmd == "report_state":
            resp = {"ok": True, "state": state.to_dict()}

        elif cmd == "reset":
            state.reset()
            resp = {"ok": True}

        elif cmd == "silence":
            state.silenced = True
            resp = {"ok": True}

        elif cmd == "unsilence":
            state.silenced = False
            resp = {"ok": True}

        elif cmd == "emit_science_packet":
            # Emit a science UDP datagram.
            # arg is JSON: {"dest_ip": "...", "dest_port": 60001, "payload_hex": "..."}
            try:
                params = json.loads(arg)
                dest_ip = params["dest_ip"]
                dest_port = int(params.get("dest_port", 60001))
                payload = bytes.fromhex(params.get("payload_hex", ""))
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(payload, (dest_ip, dest_port))
                sock.close()
                resp = {"ok": True, "sent": len(payload)}
            except Exception as e:
                resp = {"ok": False, "error": str(e)}

        else:
            resp = {"ok": False, "error": f"unknown command: {cmd!r}"}

        writer.write((json.dumps(resp) + "\n").encode())
        await writer.drain()
    except Exception as e:
        logger.warning(f"UDS handler error: {e}")
    finally:
        writer.close()


# ── Main entry point ─────────────────────────────────────────────────────────

background_tasks: set[asyncio.Task[Any]] = set()

async def main() -> None:
    loop = asyncio.get_event_loop()

    # Build server state
    state = ServerState(
        quabos=[QuaboState(quabo_index=i, module_id=MODULE_ID) for i in range(4)],
    )

    # Start UDP listeners on all four quabo command ports
    udp_transports: list[asyncio.DatagramTransport] = []
    for i, port in enumerate(CMD_PORTS):
        def protocol_factory(idx: int = i) -> QuaboProtocol:
            return QuaboProtocol(state, idx)
        transport, _ = await loop.create_datagram_endpoint(
            protocol_factory,
            local_addr=("0.0.0.0", port),
            reuse_port=True,
        )
        udp_transports.append(transport)
        logger.info(f"Listening on UDP 0.0.0.0:{port} (Q{i})")

    # Start HK emitter
    task = asyncio.create_task(hk_emitter(state))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    # Start UDS control socket
    if os.path.exists(UDS_SOCK_PATH):  # noqa: ASYNC240
        os.unlink(UDS_SOCK_PATH)
    uds_server = await asyncio.start_unix_server(
        lambda r, w: handle_control_client(r, w, state),
        path=UDS_SOCK_PATH,
    )
    os.chmod(UDS_SOCK_PATH, 0o666)
    logger.info(f"Control socket: {UDS_SOCK_PATH}")
    logger.info(f"Module ID: {MODULE_ID}, base IP: {MODULE_BASE_IP}")

    # Graceful shutdown
    stop_event = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT, stop_event.set)

    await stop_event.wait()
    logger.info("Shutting down mock_quabo server")

    uds_server.close()
    await uds_server.wait_closed()
    for t in udp_transports:
        t.close()


if __name__ == "__main__":
    asyncio.run(main())
