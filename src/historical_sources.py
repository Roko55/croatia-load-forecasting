"""Read the pinned historical ENTSO-E workbooks and construct the Croatian holiday calendar."""
import datetime as dt
from datetime import date, timedelta
import re
from xml.etree import ElementTree
import zipfile
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import pandas as pd

from audit import digest

UTC = dt.timezone.utc
ZAGREB = ZoneInfo("Europe/Zagreb")
ARCHIVE_SHA = "3a762b8dcee5567cfc84912deee1bbe5e908553d813228634bbd43e7ff0a0b20"


def read_legacy(path):
    """Read Croatian load from the pinned workbook's local-date rows and hourly columns."""
    assert digest(path) == ARCHIVE_SHA
    hour_columns = [chr(i) for i in range(70, 91)] + ['AA', 'AB', 'AC']
    observed, days, ambiguous, nonexistent, invalid = {}, set(), [], [], []
    with zipfile.ZipFile(path) as workbook, workbook.open('xl/worksheets/sheet1.xml') as sheet:
        tail = b''
        while chunk := sheet.read(1024 * 1024):
            rows = (tail + chunk).split(b'</row>')
            tail = rows.pop()
            for row in rows:
                if not re.search(rb'<c r="A\d+"[^>]* t="s"><v>25</v></c>', row):
                    continue
                cells = {key.decode(): float(value) for key, value in re.findall(
                    rb'<c r="([A-Z]+)\d+"[^>]*><v>([-\d.Ee+]+)</v></c>', row)}
                day = dt.datetime(*(int(cells[key]) for key in 'BCD'))
                assert day not in days and cells['E'] == 100
                days.add(day)
                for hour, col in enumerate(hour_columns):
                    local = day.replace(hour=hour)
                    candidates = {local.replace(tzinfo=ZAGREB, fold=fold).astimezone(UTC) for fold in (0, 1)}
                    possible = {t for t in candidates if t.astimezone(ZAGREB).replace(tzinfo=None) == local}
                    if not possible:
                        assert col not in cells
                        nonexistent.append(local.isoformat())
                    elif len(possible) > 1:
                        # One reported value has no fold label. Preserve the workbook
                        # and exclude both UTC candidates from the observed series.
                        ambiguous.append(dict(local=local.isoformat(), raw_mw=cells.get(col)))
                    elif col not in cells or cells[col] <= 0:
                        invalid.append(dict(local=local.isoformat(), raw_mw=cells.get(col)))
                    else:
                        stamp, = possible
                        assert stamp not in observed
                        observed[stamp] = cells[col]
    assert len(days) == (dt.date(2016, 1, 1) - dt.date(2006, 1, 1)).days
    assert len(ambiguous) == len(nonexistent) == 10 and len(invalid) == 1
    return pd.Series(observed, name='demand_mw').sort_index(), dict(
        days=len(days), spring_nonexistent=nonexistent, autumn_unresolved=ambiguous,
        invalid=invalid, archive_sha256=ARCHIVE_SHA)


def read_bridge(path):
    """Read Croatian UTC timestamps, demand and source coverage from the bridge workbook."""
    rows = []
    with ZipFile(path) as book:
        strings = [''.join(s.itertext()) for s in ElementTree.fromstring(book.read('xl/sharedStrings.xml'))]
        wanted = re.compile(rb'<c r="F\d+"[^>]* t="s"><v>' + str(strings.index('HR')).encode() + rb'</v></c>')
        numeric = re.compile(rb'<c r="([BGH])\d+"[^>]*><v>([-\d.Ee+]+)</v></c>')
        for sheet in book.namelist():
            if not re.fullmatch(r'xl/worksheets/sheet\d+.xml', sheet):
                continue
            with book.open(sheet) as xml:
                tail = b''
                while chunk := xml.read(1024**2):
                    batch = (tail + chunk).split(b'</row>')
                    tail = batch.pop()
                    for row in batch:
                        if not wanted.search(row):
                            continue
                        values = {key:float(value) for key,value in numeric.findall(row)}
                        rows.append((values[b'B'], values.get(b'H'), values.get(b'G')))
    data = pd.DataFrame(rows, columns=['excel_date_utc','demand_mw','coverage_pct'])
    # Excel stores fractional days as binary floats. Check snapping to hours.
    hours = data.pop('excel_date_utc') * 24
    assert ((hours-hours.round()).abs() < 1e-6).all()
    data['timestamp_utc'] = pd.Timestamp('1899-12-30', tz='UTC') + pd.to_timedelta(hours.round(), unit='h')
    data = data.set_index('timestamp_utc').sort_index()
    data = data[data.index.year.isin(range(2015,2020))]
    duplicated = data[data.index.duplicated(keep=False)]
    for _, group in duplicated.groupby(level=0):
        assert len(group.drop_duplicates()) == 1, group
    data = data[~data.index.duplicated()]
    return data


def easter_sunday(year: int) -> date:
    """Gregorian computus, valid for the supported calendar years."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    n = h + l - 7*m + 114
    return date(year, n//31, n % 31 + 1)


def croatian_holidays(year: int) -> dict[date, str]:
    """Build the Croatian public holiday calendar from 2015 onward.
    Sources: NN 136/2002 + NN 55/2008; NN 110/2019 effective 2020-01-01.
    NN 72/2025 changes memorial days only. No substitute Monday.
    Religious individual entitlements and special election days are excluded.
    """
    if not 2015 <= year <= 2100:
        raise ValueError('Calendar supports 2015..2100; future law is a scenario')
    fixed = {(1,1):'Nova godina', (1,6):'Bogojavljenje', (5,1):'Praznik rada',
             (6,22):'Dan antifasisticke borbe', (8,5):'Dan pobjede',
             (8,15):'Velika Gospa', (11,1):'Svi sveti', (12,25):'Bozic',
             (12,26):'Sveti Stjepan'}
    if year < 2020:
        fixed.update({(6,25):'Dan drzavnosti', (10,8):'Dan neovisnosti'})
    else:
        fixed.update({(5,30):'Dan drzavnosti', (11,18):'Dan sjecanja'})
    result = {date(year,*md):name for md,name in fixed.items()}
    easter = easter_sunday(year)
    # Multiple holidays may share a date (e.g. Labour Day/Easter). Binary union.
    for delta,name in [(0,'Uskrs'),(1,'Uskrsni ponedjeljak'),(60,'Tijelovo')]:
        day = easter + timedelta(days=delta)
        result[day] = result.get(day, '') + ('; ' if day in result else '') + name
    return result


def historical_holidays(year):
    # Statutory list NN 136/2002 and subsequent name-only/commemoration changes.
    fixed = [(1, 1), (1, 6), (5, 1), (6, 22), (6, 25), (8, 5),
             (8, 15), (10, 8), (11, 1), (12, 25), (12, 26)]
    easter = easter_sunday(year)
    return {dt.date(year, month, day) for month, day in fixed} | {
        easter + dt.timedelta(days=offset) for offset in (0, 1, 60)}
