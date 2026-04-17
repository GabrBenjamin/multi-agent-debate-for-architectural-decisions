import os
import getpass
from typing import Optional

def set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

def override_openai_key(openai_key: Optional[str] = None) -> None:
    """
    Override the OpenAI API key for this process only.
    - If openai_key is provided, use it.
    - Else, if MAD_OPENAI_API_KEY is set, use that.
    - Else, raise an error. We deliberately do NOT fall back to any pre-existing
      OPENAI_API_KEY to avoid using the default environment variable.
    """
    key = openai_key or os.environ.get("MAD_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No OpenAI key provided. Pass --openai-key or set MAD_OPENAI_API_KEY.")
    os.environ["OPENAI_API_KEY"] = key
