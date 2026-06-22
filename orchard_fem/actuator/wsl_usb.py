# -*- coding: utf-8 -*-
"""Auto-attach a USB serial adapter into WSL2 before connecting to the rig.

WSL2 does not expose host USB devices by default; ``usbipd-win`` forwards them.
``usbipd attach`` needs **no** administrator rights (only the one-time ``bind``
does), and WSL can invoke the Windows ``usbipd.exe`` over interop — so the rig
linkage can attach the cylinder's serial adapter automatically right before
opening the port, instead of the user running PowerShell by hand every session.

One-time prerequisites (done once, survive reboots):
  * Windows (admin):  ``usbipd bind --busid <X>``  (marks the adapter shareable)
  * WSL: load the serial driver so the attached device gets a ``/dev/ttyUSB*``
    node — persist it with::

        echo ch341 | sudo tee /etc/modules-load.d/ch341.conf   # CH340/CH341
        # (cp210x for Silicon Labs, ftdi_sio for FTDI adapters)

This module is best-effort: on any failure it returns an actionable message and
the caller falls back to its normal "open the port" path.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time

# Substrings that mark a USB-serial adapter row in ``usbipd list``.
_SERIAL_HINTS = ("SERIAL", "CH340", "CH341", "CP210", "FTDI", "(COM")
_USBIPD_CANDIDATES = (
    "/mnt/c/Program Files/usbipd-win/usbipd.exe",
    "/mnt/c/Program Files (x86)/usbipd-win/usbipd.exe",
)


def is_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _find_usbipd() -> str | None:
    found = shutil.which("usbipd.exe")
    if found:
        return found
    return next((c for c in _USBIPD_CANDIDATES if os.path.exists(c)), None)


def _serial_ports() -> list[str]:
    return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))


def _try_load_serial_drivers() -> None:
    """Best-effort, non-interactive ``modprobe`` of the usb-serial drivers.

    Succeeds only with passwordless sudo (or when already root); otherwise it
    fails fast and the caller falls back to asking the user to load / persist
    the module.  Never blocks on a password prompt (``sudo -n``).
    """
    for mod in ("ch341", "cp210x", "ftdi_sio"):
        try:
            subprocess.run(["sudo", "-n", "modprobe", mod],
                           capture_output=True, timeout=8)
        except Exception:  # noqa: BLE001
            return


def _ensure_openable(node: str) -> tuple[bool, str]:
    """Given an existing tty *node*, ensure it's readable/writable (fix perms if able)."""
    if os.access(node, os.R_OK | os.W_OK):
        return True, f"{node} ready."
    _try_fix_permissions(node)
    if os.access(node, os.R_OK | os.W_OK):
        return True, f"{node} ready (permissions fixed)."
    return False, (f"{node} exists but is not accessible (WSL creates it root-only). "
                   f"Run:  sudo chmod a+rw {node}")


def _try_fix_permissions(port: str) -> None:
    """Best-effort ``sudo -n chmod`` so a usbip node (root:root 0600 in WSL) opens.

    WSL has no udev rule to put the node in the ``dialout`` group, so it lands
    root-only; with passwordless sudo we relax it, otherwise the caller reports
    the manual ``chmod`` to run.  Never prompts (``sudo -n``).
    """
    try:
        subprocess.run(["sudo", "-n", "chmod", "a+rw", port],
                       capture_output=True, timeout=8)
    except Exception:  # noqa: BLE001
        pass


def ensure_usb_serial(
    port: str | None = None,
    *,
    busid: str | None = None,
    timeout_s: float = 6.0,
    log=print,
) -> tuple[bool, str]:
    """Ensure a USB-serial node exists in WSL, auto-attaching it via usbipd.

    Returns ``(ok, message)``.  No-op success when not under WSL, or when *port*
    (or any ``/dev/ttyUSB*``) is already present.  When *busid* is given only
    that bus id is considered; otherwise the first serial-looking adapter in
    ``usbipd list`` is used.
    """
    # Already have a node? Make sure it actually opens (WSL usbip nodes are
    # root:root 0600), fixing perms if we can; only attach if there's no node.
    existing = port if (port and os.path.exists(port)) else next(iter(_serial_ports()), None)
    if existing:
        return _ensure_openable(existing)
    if not is_wsl():
        return False, f"{port or 'serial port'} not found (not WSL — check cable / port name)."

    usbipd = _find_usbipd()
    if usbipd is None:
        return False, ("usbipd.exe not found — install usbipd-win on Windows, then run "
                       "`usbipd bind --busid <X>` once (admin).")

    try:
        listing = subprocess.run(
            [usbipd, "list"], capture_output=True, text=True, timeout=15,
        ).stdout.replace("\r", "")
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not run usbipd list: {exc}"

    target: str | None = None
    already_attached = False
    for line in listing.splitlines():
        upper = line.upper()
        if not (line[:1].isdigit() and any(h in upper for h in _SERIAL_HINTS)):
            continue
        bid = line.split()[0]
        if busid and bid != busid:
            continue
        if "NOT SHARED" in upper:
            return False, (f"USB serial {bid} is not shared — run once as admin on "
                           f"Windows:  usbipd bind --busid {bid}")
        target = bid
        already_attached = "ATTACHED" in upper
        break

    if target is None:
        return False, "No USB-serial adapter found in `usbipd list` (check the cable)."

    if not already_attached:
        log(f"Attaching USB serial {target} to WSL via usbipd…")
        res = subprocess.run(
            [usbipd, "attach", "--wsl", "--busid", target],
            capture_output=True, text=True, timeout=30,
        )
        if res.returncode != 0:
            return False, f"usbipd attach failed: {(res.stderr or res.stdout).strip()}"

    tried_modprobe = False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        node = port if (port and os.path.exists(port)) else next(iter(_serial_ports()), None)
        if node:
            return _ensure_openable(node)
        if not tried_modprobe:           # node missing → try to load the driver once
            _try_load_serial_drivers()
            tried_modprobe = True
        time.sleep(0.4)

    return False, ("Device attached but no /dev/ttyUSB* appeared — load the serial "
                   "driver:  sudo modprobe ch341   (persist via /etc/modules-load.d/).")
