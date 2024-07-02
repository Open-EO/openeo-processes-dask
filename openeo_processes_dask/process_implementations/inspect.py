import logging
from typing import Any, Optional

__all__ = ["inspect"]

logger = logging.getLogger(__name__)


def inspect(
    data: Any,
    message: Optional[str] = "",
    code: Optional[str] = "User",
    level: Optional[str] = "info",
) -> Any:
    if level.lower() == "info":
        logger.info(f"{code}: {message} {data}.")

    elif level.lower() == "warning":
        logger.warning(f"{code}: {message} {data}.")

    elif level.lower() == "error":
        logger.error(f"{code}: {message} {data}.")

    elif level.lower() == "debug":
        logger.debug(f"{code}: {message} {data}.")

    return data
