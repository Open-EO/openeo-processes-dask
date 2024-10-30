from openeo_processes_dask.process_implementations.dates import (
    date_between,
    date_difference,
    date_shift,
)


def test_date_between():
    assert not date_between(x="2020-01-01", min="2021-01-01", max="2022-01-01")


def test_date_difference():
    assert (
        date_difference(date1="2020-01-01T00:00:00.0Z", date2="2020-01-01T00:00:15.5Z")
        == 15.5
    )
    assert (
        date_difference(date1="2020-01-01T00:00:00Z", date2="2020-01-01T01:00:00+01:00")
        == 0
    )
    assert date_difference(date1="2020-01-02", date2="2020-01-01") == -86400
    assert date_difference(date1="2020-01-02", date2="2020-01-01", unit="day") == -1


def test_date_shift():
    month_shift = date_shift(date="2020-02-01T17:22:45Z", value=6, unit="month")
    assert month_shift == "2020-08-01T17:22:45Z"

    day_shift = date_shift(date="2021-03-31T00:00:00+02:00", value=-7, unit="day")
    assert day_shift == "2021-03-24T00:00:00+02:00"

    year_shift = date_shift(date="2020-02-29T17:22:45Z", value=1, unit="year")
    assert year_shift == "2021-02-28T17:22:45Z"

    month_shift = date_shift(date="2020-01-31", value=1, unit="month")
    assert month_shift == "2020-02-29"

    second_shift = date_shift(date="2016-12-31T23:59:59Z", value=1, unit="second")
    assert second_shift == "2017-01-01T00:00:00Z"

    millisecond_shift = date_shift(
        date="2018-12-31T17:22:45Z", value=1150, unit="millisecond"
    )
    assert millisecond_shift == "2018-12-31T17:22:46.150Z"

    hour_shift = date_shift(date="2018-01-01", value=25, unit="hour")
    assert hour_shift == "2018-01-02"

    hour_shift = date_shift(date="2018-01-01", value=-1, unit="hour")
    assert hour_shift == "2017-12-31"
