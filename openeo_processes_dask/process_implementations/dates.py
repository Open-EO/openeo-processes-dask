from typing import Optional

import numpy as np

__all__ = [
    "date_between",
    "date_difference",
    "date_shift",
]


def datetime_from_str(date: str):
    daytime = np.datetime64(date)
    return daytime


def date_between(
    x: str, min: str, max: str, exclude_max: bool = False
) -> Optional[bool]:
    x = datetime_from_str(x)
    min = datetime_from_str(min)
    max = datetime_from_str(max)
    if exclude_max:
        return bool((x >= min) and (x < max))
    else:
        return bool((x >= min) and (x <= max))


def date_difference(date1: str, date2: str, unit: Optional[str] = "second") -> float:
    date1 = datetime_from_str(date1)
    date2 = datetime_from_str(date2)
    units = {
        "millisecond": 1,
        "second": 1000,
        "minute": 1000 * 60,
        "hour": 1000 * 60 * 60,
        "day": 1000 * 60 * 60 * 24,
        "week": 1000 * 60 * 60 * 24 * 7,
        "month": "M",
        "year": "Y",
    }
    if unit in units:
        unit = units[unit]
    if unit in ["M", "Y"]:
        return float(
            (
                date2.astype(f"datetime64[{unit}]")
                - date1.astype(f"datetime64[{unit}]")
            ).astype(float)
        )
    else:
        # we do this, so the examples are fulfilled:
        # date_difference(date1 = "2020-01-01T00:00:00.0Z", date2 = "2020-01-01T00:00:15.5Z") -> 15.5
        return (
            float(
                (
                    date2.astype(f"datetime64[ms]") - date1.astype(f"datetime64[ms]")
                ).astype(float)
            )
            / unit
        )


def date_shift(date: str, value: int, unit: str) -> str:
    if date.endswith("Z"):
        end = "Z"
    elif "+" in date:
        end = "+" + date.split("+")[-1]
        date = date.split("+")[0]
    else:
        end = ""
    units = {
        "millisecond": "ms",
        "second": "s",
        "minute": "m",
        "hour": "h",
        "day": "D",
        "week": "W",
        "month": "M",
        "year": "Y",
    }
    if unit in units:
        unit = units[unit]
    if unit in ["M", "Y"]:
        if len(date) > 7:
            date_M = np.datetime64(date, "M")
            day = (
                int(
                    (np.datetime64(date, "D") - date_M.astype("datetime64[D]")).astype(
                        int
                    )
                )
                + 1
            )
            if " " in date:
                time = "T" + date.split(" ")[-1]
            elif "T" in date:
                time = "T" + date.split("T")[-1]
            else:
                time = ""
            new_date = str(date_M + np.timedelta64(value, unit))
            if day in [29, 30, 31]:
                for i in range(3):
                    try:
                        new_daytime = f"{new_date}-{day-i}"
                        new_daytime_numpy = np.datetime64(new_daytime)
                        result = f"{new_daytime}{time}"
                        return result
                    except:
                        pass
            elif int(day) < 10:
                new_daytime = f"{new_date}-0{day}{time}"
            else:
                new_daytime = f"{new_date}-{day}T{time}"
            new_daytime_numpy = np.datetime64(new_daytime)
            return new_daytime

        date = datetime_from_str(date)
        return str(date_M + np.timedelta64(value, unit))

    date = datetime_from_str(date)
    if unit in ["ms"]:
        result = str((date + np.timedelta64(value, unit)).astype(f"datetime64[{unit}]"))
    else:
        result = str((date + np.timedelta64(value, unit)).astype(date.dtype))
    return result + end
