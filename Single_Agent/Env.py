import getpass
import os


def set_env(var: str) -> str:
    """Return an existing value or request one for the current process."""
    value = os.environ.get(var)
    if value:
        return value

    value = getpass.getpass(f"{var}: ").strip()
    if not value:
        raise RuntimeError(f"{var} is required.")

    os.environ[var] = value
    return value
