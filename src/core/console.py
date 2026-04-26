from __future__ import annotations


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_ok(message: str) -> None:
    print(f"[OK] {message}")


def print_warning(message: str) -> None:
    print(f"[WARN] {message}")


def print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")
