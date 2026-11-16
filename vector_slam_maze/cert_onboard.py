"""Automated Vector TLS session certificate & sdk_config.ini onboarding."""

from __future__ import annotations

import argparse
import configparser
import io
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

DEFAULT_BOT_SDK_INFO = Path(os.environ.get("WIRE_POD_BOT_SDK_INFO", "/home/pi/wire-pod/chipper/jdocs/botSdkInfo.json"))
DEFAULT_SESSION_CERTS_DIR = Path(os.environ.get("WIRE_POD_SESSION_CERTS", "/home/pi/wire-pod/chipper/session-certs"))
DEFAULT_SDK_DIR = Path(os.environ.get("ANKI_VECTOR_SDK_DIR", "/home/pi/.anki_vector"))
MIN_CERT_CHARS = 64
MAX_BOT_ENTRIES = 256


def _atomic_write_text(path: Path, body: str, mode: int = 0o600) -> None:
    """Write UTF-8 file atomically with explicit permissions via sibling temp file."""
    assert path is not None and isinstance(body, str)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.fchmod(fd, mode)
        os.write(fd, body.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp, path)
    assert path.is_file()


def load_wirepod_sdk_entry(bot_sdk_info_path: Path, serial: str) -> Optional[Dict[str, Any]]:
    """Extract wire-pod registration entry (ip_address, guid, activated) for serial."""
    assert bot_sdk_info_path is not None and serial, "botSdkInfo path and serial required"
    raw = json.loads(bot_sdk_info_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "botSdkInfo.json must decode to JSON object"
    norm = serial.strip().lower()
    if norm in raw and isinstance(raw[norm], dict):
        return dict(raw[norm])
    for container in ("robots", "bots", "esns"):
        sub = raw.get(container)
        if isinstance(sub, dict) and isinstance(sub.get(norm), dict):
            return dict(sub[norm])
        if isinstance(sub, list):
            for item in sub[:MAX_BOT_ENTRIES]:
                if isinstance(item, dict) and str(item.get("esn") or item.get("serial") or "").strip().lower() == norm:
                    merged = dict(item)
                    if not merged.get("guid") and raw.get("global_guid"):
                        merged["guid"] = str(raw["global_guid"])
                    return merged
    return None


def read_cached_session_cert(session_certs_dir: Path, serial: str) -> Optional[str]:
    """Return non-empty PEM string from wire-pod session-certs cache if present."""
    assert session_certs_dir is not None and serial, "session_certs_dir and serial required"
    for cand in (session_certs_dir / serial, session_certs_dir / f"{serial}.cert"):
        if cand.is_file():
            body = cand.read_text(encoding="utf-8", errors="replace").strip()
            if "BEGIN CERTIFICATE" in body and len(body) >= MIN_CERT_CHARS:
                return body + "\n"
    return None


def fetch_peer_pem_cert(ip: str, timeout_s: float = 2.0) -> Tuple[Optional[str], Optional[str]]:
    """Probe Vector TCP :443 and extract leaf TLS peer certificate in PEM format."""
    assert ip and timeout_s > 0.0, "valid IP and positive timeout required"
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((ip, 443), timeout=timeout_s) as raw_sock:
            with ctx.wrap_socket(raw_sock, server_hostname=ip) as tls_sock:
                der = tls_sock.getpeercert(binary_form=True)
        if not der:
            return (None, "empty_peer_cert")
        return (ssl.DER_cert_to_PEM_cert(der), None)
    except (OSError, TimeoutError, ssl.SSLError, ValueError) as exc:
        return (None, f"{type(exc).__name__}:{exc}")


def extract_cert_common_name(pem: str) -> Optional[str]:
    """Decode subject commonName (e.g. Vector-DEMO) from X.509 PEM certificate."""
    assert isinstance(pem, str) and "BEGIN CERTIFICATE" in pem, "valid PEM required"
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", encoding="utf-8", delete=False) as tf:
            tf.write(pem)
            tmp_path = Path(tf.name)
        decoder = getattr(getattr(ssl, "_ssl", None), "_test_decode_cert", None)
        if callable(decoder):
            info = decoder(str(tmp_path))
            for rdn in info.get("subject", ()):
                for key, val in rdn:
                    if key == "commonName" and val:
                        return str(val)
        proc = subprocess.run(["openssl", "x509", "-in", str(tmp_path), "-noout", "-subject", "-nameopt", "RFC2253"], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            for part in proc.stdout.strip().replace("subject=", "").split(","):
                if part.strip().startswith("CN="):
                    return part.strip()[3:]
        return None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def upsert_sdk_config_ini(
    config_path: Path, serial: str, cert_path: Path, ip: str, name: str, guid: str
) -> None:
    """Update [serial] section in sdk_config.ini atomically with 0600 mode preserving peers."""
    assert serial and ip and name, "serial, ip, and name required"
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.optionxform = str  # type: ignore[assignment]
    if config_path.is_file():
        cfg.read(config_path, encoding="utf-8")
    cfg[serial] = {"cert": str(cert_path), "ip": ip, "name": name, "guid": guid}
    assert serial in cfg.sections()
    buf = io.StringIO()
    cfg.write(buf)
    _atomic_write_text(config_path, buf.getvalue(), mode=0o600)


def _resolve_cert_material(
    ip: str, serial: str, session_certs_dir: Path, timeout_s: float, allow_live_fallback: bool = False
) -> Tuple[Optional[str], str, Optional[str]]:
    """Prefer wire-pod session-certs cache, verify live :443 matches, gate uncached live fallback."""
    assert serial and session_certs_dir is not None
    cached = read_cached_session_cert(session_certs_dir, serial)
    live_pem, tls_err = (fetch_peer_pem_cert(ip, timeout_s=timeout_s) if ip else (None, "missing_ip"))
    if cached and live_pem:
        if ssl.PEM_cert_to_DER_cert(cached) != ssl.PEM_cert_to_DER_cert(live_pem):
            return (None, "cert_mismatch_vs_wire_pod_cache", "live :443 cert differs from wire-pod session-certs cache")
        return (cached, "session_certs_verified_live", None)
    if cached:
        return (cached, "session_certs_cache", tls_err)
    if not live_pem:
        return (None, "tls_unreachable", tls_err)
    if not allow_live_fallback:
        return (None, "live_cert_unverified", "no wire-pod session cert cached; pass --allow-live-cert-fallback to trust live :443 cert")
    return (live_pem, "tls_443_fallback", None)


def onboard_vector_cert(
    serial: str,
    bot_sdk_info_path: Path = DEFAULT_BOT_SDK_INFO,
    session_certs_dir: Path = DEFAULT_SESSION_CERTS_DIR,
    sdk_dir: Path = DEFAULT_SDK_DIR,
    sdk_config_path: Optional[Path] = None,
    robot_name: Optional[str] = None,
    timeout_s: float = 2.0,
    dry_run: bool = False,
    allow_live_fallback: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    """Onboard Vector .cert and 0600 [serial] sdk_config.ini entry using CN name resolution."""
    assert serial and timeout_s > 0.0, "serial and timeout_s > 0 required"
    cfg_path = sdk_config_path or (sdk_dir / "sdk_config.ini")
    if not bot_sdk_info_path.is_file():
        return (2, {"ok": False, "error": "bot_sdk_info_missing", "path": str(bot_sdk_info_path), "sdk_config_mutated": False})
    entry = load_wirepod_sdk_entry(bot_sdk_info_path, serial)
    if entry is None:
        return (2, {"ok": False, "error": "serial_not_registered_in_wire_pod", "serial": serial, "sdk_config_mutated": False})
    ip, guid, activated = str(entry.get("ip_address") or entry.get("ip") or ""), str(entry.get("guid") or ""), bool(entry.get("activated", False))
    pem, source, detail = _resolve_cert_material(ip, serial, session_certs_dir, timeout_s, allow_live_fallback=allow_live_fallback)
    if not pem:
        return (3, {"ok": False, "error": source, "serial": serial, "ip": ip, "activated": activated, "guid_present": bool(guid), "tls_error": detail, "sdk_config_mutated": False})
    cn = extract_cert_common_name(pem)
    if robot_name and cn and cn.startswith("Vector-") and robot_name != cn:
        return (3, {"ok": False, "error": "robot_name_mismatch_vs_cert_cn", "serial": serial, "robot_name": robot_name, "cert_cn": cn, "sdk_config_mutated": False})
    name = robot_name or (cn if (cn and cn.startswith("Vector-")) else None)
    if not name:
        return (2, {"ok": False, "error": "robot_name_unresolvable", "serial": serial, "cert_cn": cn, "sdk_config_mutated": False})
    cert_path = sdk_dir / f"{name}-{serial}.cert"
    if not dry_run:
        _atomic_write_text(cert_path, pem, mode=0o644)
        upsert_sdk_config_ini(cfg_path, serial, cert_path, ip, name, guid)
    return (0, {"ok": True, "serial": serial, "ip": ip, "name": name, "cert_cn": cn, "activated": activated, "source": source, "cert_path": str(cert_path), "sdk_config_path": str(cfg_path), "dry_run": dry_run, "sdk_config_mutated": not dry_run})


def cmd_onboard_cert(args: argparse.Namespace) -> int:
    """CLI handler for wire-pod botSdkInfo + :443 TLS cert onboarding."""
    assert args.serial, "serial required"
    rc, payload = onboard_vector_cert(
        serial=str(args.serial),
        bot_sdk_info_path=Path(args.bot_sdk_info),
        session_certs_dir=Path(args.session_certs_dir),
        sdk_dir=Path(args.sdk_dir),
        sdk_config_path=Path(args.sdk_config) if args.sdk_config else None,
        robot_name=args.robot_name,
        timeout_s=float(args.timeout_s),
        dry_run=bool(args.dry_run),
        allow_live_fallback=bool(getattr(args, "allow_live_cert_fallback", False)),
    )
    print(json.dumps(payload, indent=2))
    assert rc in (0, 2, 3)
    return rc
