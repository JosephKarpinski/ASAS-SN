"""
cosmos_desi_match.py
--------------------
Cross-match the COSMOS-Web (COSMOS2025) galaxy catalogue with DESI DR1 by sky position and write a list of
DESI TARGETIDs that unite_desi_dr1.py can process (set TARGETID_FILE there).

Run from VSCode / Jupyter (paste into a cell) or a terminal:  python cosmos_desi_match.py

Steps
  1. COSMOS side   SOURCE = 'vizier' : COSMOS2025 table (J/A+A/704/A339) from VizieR: astroquery on the Harvard
                                       mirror, then the IDIA mirror (CDS Strasbourg is down: cybersecurity incident)
                   SOURCE = 'file'   : a table you downloaded yourself (COSMOS_FILE: .csv, .fits, .ecsv, ...)
                                       with RA / Dec columns (and, if present, an ID and a photometric redshift)
  2. DESI side     Data Lab (NOIRLab, USA) table desi_dr1.zpix, restricted to the COSMOS sky box
  3. match         nearest COSMOS source to each DESI target, kept if closer than MATCH_ARCSEC

Outputs (in OUT_DIR)
    cosmos_desi_matches.csv     every match: COSMOS id / RA / Dec / photo-z, DESI targetid / survey / program /
                                healpix / z / zwarn / spectype, separation, and z_desi - z_phot
    cosmos_desi_targetids.csv   the DESI targets unite_desi_dr1.py can process (spectype, ZWARN and z cuts)

Then in unite_desi_dr1.py:   TARGETID_FILE = 'cosmos_desi_targetids.csv'   (and N_SPECTRA / START as usual)

Needs: numpy, pandas, astropy, requests (and optionally astro-datalab).
"""
import io
import os
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------------ settings
SOURCE = 'vizier'                 # 'vizier' or 'file'
COSMOS_FILE = ''                  # SOURCE = 'file': path to the table you downloaded
VIZIER_TAP = []                   # TAP services to try after the servers above, e.g. ['https://tapvizier.cds.unistra.fr/TAPVizieR/tap']
VIZIER_SERVERS = ['vizier.cfa.harvard.edu',       # Harvard mirror
                  'vizier.nao.ac.jp',             # Japan
                  'vizier.china-vo.org',          # China
                  'vizier.ast.cam.ac.uk',         # Cambridge UK
                  'vizier.hia.nrc.ca']            # Canada
                  # astroquery only accepts these names (plus 'vizier.cds.unistra.fr' and 'vizier.iucaa.ernet.in');
                  # add 'vizier.cds.unistra.fr' at the end once CDS is back
VIZIER_TIMEOUT = 120              # seconds to wait on each server before trying the next
VIZIER_CATALOG = 'J/A+A/704/A339'     # COSMOS2025 (Shuntov et al. 2025, A&A 704, A339)
VIZIER_TABLE = ''                 # '' = list the catalogue's tables and take the one with the most rows
RA_COL = DEC_COL = ID_COL = ZPHOT_COL = ''    # '' = guess from the column names (the guess is printed)

BOX = None                        # (ra_min, ra_max, dec_min, dec_max) in degrees; None = from the COSMOS table
BOX_FALLBACK = (149.3, 151.0, 1.4, 3.0)   # used by SOURCE = 'vizier' (the whole COSMOS field)
MATCH_ARCSEC = 1.0                # DESI fibre / target positions are good to well under 1 arcsec
SPECTYPE = 'GALAXY'               # which Redrock class unite_desi_dr1.py processes
REQUIRE_ZWARN0 = True
PRIMARY_ONLY = True
Z_RANGE = None                    # e.g. (0.01, 0.45): keep Halpha/[NII]/[SII] inside DESI; None = no cut
OUT_DIR = '.'

ZPIX_TABLE = 'desi_dr1.zpix'
DATALAB_QUERY_URL = 'https://datalab.noirlab.edu/query/query'
PRIMARY_COLUMNS = ['zcat_primary', 'main_primary', 'sv_primary', 'specprimary']
DESI_POS_COLUMNS = [('target_ra', 'target_dec'), ('mean_fiber_ra', 'mean_fiber_dec'), ('ra', 'dec')]


# ------------------------------------------------------------------ COSMOS side
def _tap(url, adql, timeout=300):
    r = requests.post(url + '/sync', data={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv',
                                           'QUERY': adql}, timeout=timeout)
    r.raise_for_status()
    if r.text.lstrip().startswith('<'):
        raise RuntimeError('TAP service returned an error: ' + r.text[:300].replace('\n', ' '))
    return pd.read_csv(io.StringIO(r.text))


def _guess(cols, wanted, what, override=''):
    if override:
        return override
    low = {c.lower(): c for c in cols}
    for w in wanted:
        if w in low:
            return low[w]
    for c in cols:
        if any(w in c.lower() for w in wanted):
            return c
    raise SystemExit(f'Could not guess the {what} column from: {list(cols)}\n'
                     f'Set it by hand (RA_COL / DEC_COL / ID_COL / ZPHOT_COL).')


def load_vizier_http(server):
    """Ask a VizieR server directly (ASU interface) for the id/RA/Dec of the COSMOS2025 photometry table inside
    the box, and parse the VOTable ourselves (astroquery chokes on the odd reply the Harvard mirror gives)."""
    import io, re, requests
    from astropy.io.votable import parse_single_table
    b = BOX or BOX_FALLBACK
    table = VIZIER_TABLE or VIZIER_CATALOG + '/phot'
    params = {'-source': table, '-out': 'Id,RAJ2000,DEJ2000', '-out.max': 'unlimited',
              'RAJ2000': f'{b[0]}..{b[1]}', 'DEJ2000': f'{b[2]}..{b[3]}'}
    print(f'  asking {server} for {table} in RA {b[0]}-{b[1]}, Dec {b[2]}-{b[3]} ...')
    r = requests.get(f'https://{server}/viz-bin/votable', params=params, timeout=(30, 600))
    r.raise_for_status()
    txt = r.text
    txt = re.sub(r'<INFO name="QUERY_STATUS"[^\n]*</INFO>', '', txt)      # malformed line in the reply
    txt = re.sub(r'<INFO name="Warning"[^\n]*/>', '', txt)
    if '<TABLE' not in txt:
        raise RuntimeError('reply holds no table: ' + txt[-300:].replace('\n', ' '))
    tab = parse_single_table(io.BytesIO(txt.encode('utf-8'))).to_table()
    df = tab.to_pandas()
    if df.empty:
        raise RuntimeError('table came back with no rows')
    print(f'  {len(df)} rows')
    return df.rename(columns={'Id': 'cosmos_id', 'RAJ2000': 'ra', 'DEJ2000': 'dec'})


def load_vizier_astroquery(server):
    """The catalogue through astroquery's VizieR interface on one VizieR web server (the Harvard mirror serves
    the same catalogues as CDS)."""
    from astroquery.vizier import Vizier, conf
    conf.server = server
    conf.timeout = VIZIER_TIMEOUT
    info = Vizier(row_limit=1).query_constraints(catalog=VIZIER_CATALOG)
    if info is None or len(info) == 0:
        raise RuntimeError(f'catalogue {VIZIER_CATALOG} not found on {server}')
    names = [str(k) for k in info.keys()]
    print('  tables:', ', '.join(names))
    tname = VIZIER_TABLE or names[0]
    t0 = info[tname]
    cols = list(t0.colnames)
    ra = _guess(cols, ['ra', 'raj2000', 'ra_deg'], 'RA', RA_COL)
    dec = _guess(cols, ['dec', 'dej2000', 'dec_deg'], 'Dec', DEC_COL)
    idc = _guess(cols, ['id', 'objid', 'name', 'seq', 'recno'], 'ID', ID_COL)
    zph = ZPHOT_COL or next((c for c in cols if any(k in c.lower() for k in ('zphot', 'zpdf', 'zml', 'z_ph', 'photoz'))), '')
    b = BOX or BOX_FALLBACK
    want = list(dict.fromkeys([idc, ra, dec] + ([zph] if zph else [])))
    v = Vizier(columns=want, row_limit=-1, column_filters={ra: f'>{b[0]} & <{b[1]}', dec: f'>{b[2]} & <{b[3]}'})
    res = v.query_constraints(catalog=tname)
    if res is None or len(res) == 0:
        raise RuntimeError(f'no rows returned by {server} for table {tname}')
    tab = res[0]
    df = tab.to_pandas()
    return df.rename(columns={idc: 'cosmos_id', ra: 'ra', dec: 'dec', **({zph: 'z_phot'} if zph else {})})


def load_vizier():
    last = None
    for server in VIZIER_SERVERS:                  # 1) VizieR web interface through astroquery
        try:
            print(f'VizieR (direct): {server}')
            return load_vizier_http(server)
        except Exception as e:
            last = e
            print(f'  direct query failed: {type(e).__name__}: {e}')
        try:
            print(f'VizieR (astroquery): {server}')
            return load_vizier_astroquery(server)
        except Exception as e:
            last = e
            print(f'  failed: {type(e).__name__}: {e}')
    for tap in VIZIER_TAP:
        try:
            print(f'VizieR TAP: {tap}')
            tabs = _tap(tap, "SELECT table_name FROM TAP_SCHEMA.tables WHERE table_name LIKE '"
                        + VIZIER_CATALOG + "%'")
            names = list(tabs.iloc[:, 0])
            if not names:
                raise RuntimeError(f'catalogue {VIZIER_CATALOG} not found on this service')
            print('  tables:', ', '.join(names))
            if VIZIER_TABLE:
                table = VIZIER_TABLE
            else:
                counts = {n: int(_tap(tap, f'SELECT COUNT(*) FROM "{n}"').iloc[0, 0]) for n in names}
                table = max(counts, key=counts.get)
                print('  row counts:', counts, '-> using', table)
            cols = list(_tap(tap, f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name='{table}'").iloc[:, 0])
            print('  columns:', ', '.join(cols[:60]) + (' ...' if len(cols) > 60 else ''))
            ra = _guess(cols, ['ra', 'raj2000', 'ra_deg'], 'RA', RA_COL)
            dec = _guess(cols, ['dec', 'dej2000', 'dec_deg'], 'Dec', DEC_COL)
            idc = _guess(cols, ['id', 'objid', 'name', 'seq', 'recno'], 'ID', ID_COL)
            zph = ZPHOT_COL or next((c for c in cols if any(k in c.lower() for k in ('zphot', 'zpdf', 'zml', 'z_ph', 'photoz'))), '')
            b = BOX or BOX_FALLBACK
            sel = ', '.join(f'"{c}"' for c in dict.fromkeys([idc, ra, dec] + ([zph] if zph else [])))
            q = (f'SELECT {sel} FROM "{table}" WHERE "{ra}" BETWEEN {b[0]} AND {b[1]} '
                 f'AND "{dec}" BETWEEN {b[2]} AND {b[3]}')
            df = _tap(tap, q, timeout=900)
            return df.rename(columns={idc: 'cosmos_id', ra: 'ra', dec: 'dec', **({zph: 'z_phot'} if zph else {})})
        except Exception as e:
            last = e
            print(f'  failed: {type(e).__name__}: {e}')
    raise SystemExit(f'No VizieR TAP service worked ({last}).  Download the table yourself and use SOURCE = "file".')


def load_file():
    from astropy.table import Table
    if not COSMOS_FILE or not os.path.exists(COSMOS_FILE):
        raise SystemExit(f'COSMOS_FILE not found: {COSMOS_FILE!r}')
    if COSMOS_FILE.lower().endswith('.csv'):
        df = pd.read_csv(COSMOS_FILE)
    else:
        t = Table.read(COSMOS_FILE)
        df = t[[c for c in t.colnames if getattr(t[c], 'ndim', 1) == 1]].to_pandas()
    cols = list(df.columns)
    print('columns:', ', '.join(cols[:60]) + (' ...' if len(cols) > 60 else ''))
    ra = _guess(cols, ['ra', 'raj2000', 'ra_deg'], 'RA', RA_COL)
    dec = _guess(cols, ['dec', 'dej2000', 'dec_deg'], 'Dec', DEC_COL)
    idc = ID_COL or next((c for c in cols if c.lower() in ('id', 'objid', 'id_cosmos2025', 'name')), None)
    zph = ZPHOT_COL or next((c for c in cols if any(k in c.lower() for k in ('zphot', 'zpdf', 'zml', 'z_ph', 'photoz'))), '')
    out = pd.DataFrame({'cosmos_id': df[idc] if idc else np.arange(len(df)), 'ra': df[ra].astype(float),
                        'dec': df[dec].astype(float)})
    if zph:
        out['z_phot'] = pd.to_numeric(df[zph], errors='coerce')
    return out


# ------------------------------------------------------------------ DESI side
def datalab_sql(sql):
    print(f'Data Lab query:\n  {sql}\n')
    try:
        from dl import queryClient as qc
        text = qc.query(sql=sql, fmt='csv')
    except ImportError:
        r = requests.get(DATALAB_QUERY_URL, params={'sql': sql, 'ofmt': 'csv', 'async': 'false'}, timeout=300)
        text = r.text
    head = str(text).lstrip().split('\n', 1)[0]
    if ',' not in head or head.lower().startswith(('error', '<')):
        raise SystemExit('Data Lab did not return a table. Its reply began:\n  ' + str(text)[:400])
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_desi(box):
    have = set(datalab_sql(f'SELECT * FROM {ZPIX_TABLE} LIMIT 1').columns)
    pos = next(((a, d) for a, d in DESI_POS_COLUMNS if a in have and d in have), None)
    if pos is None:
        raise SystemExit(f'{ZPIX_TABLE} has none of the position columns {DESI_POS_COLUMNS}; found: {sorted(have)}')
    print('DESI positions from', pos)
    cols = [c for c in ['targetid', 'survey', 'program', 'healpix', 'z', 'zwarn', 'spectype', 'subtype', pos[0], pos[1]]
            if c in have]
    cond = [f'{pos[0]} BETWEEN {box[0]} AND {box[1]}', f'{pos[1]} BETWEEN {box[2]} AND {box[3]}']
    if 'coadd_fiberstatus' in have:
        cond.append('coadd_fiberstatus = 0')
    if PRIMARY_ONLY:
        prim = next((c for c in PRIMARY_COLUMNS if c in have), None)
        if prim:
            cond.append(f'CAST({prim} AS integer) = 1')
    df = datalab_sql(f'SELECT {", ".join(cols)} FROM {ZPIX_TABLE} WHERE ' + ' AND '.join(cond))
    return df.rename(columns={pos[0]: 'desi_ra', pos[1]: 'desi_dec', 'z': 'z_desi'}).drop_duplicates('targetid')


# ------------------------------------------------------------------ match
def match(cosmos, desi, arcsec):
    c = SkyCoord(cosmos['ra'].to_numpy() * u.deg, cosmos['dec'].to_numpy() * u.deg)
    d = SkyCoord(desi['desi_ra'].to_numpy() * u.deg, desi['desi_dec'].to_numpy() * u.deg)
    idx, sep, _ = d.match_to_catalog_sky(c)
    ok = sep.arcsec <= arcsec
    m = desi[ok].reset_index(drop=True).copy()
    cc = cosmos.iloc[idx[ok]].reset_index(drop=True)
    cc = cc.rename(columns={'ra': 'cosmos_ra', 'dec': 'cosmos_dec'})  # cosmos_id and z_phot keep their names
    out = pd.concat([cc, m], axis=1)
    out['sep_arcsec'] = sep.arcsec[ok]
    if 'z_phot' in out:
        out['dz'] = out['z_desi'] - out['z_phot']
    return out.sort_values('sep_arcsec').reset_index(drop=True)


def main():
    cosmos = load_vizier() if SOURCE == 'vizier' else load_file()
    cosmos = cosmos.dropna(subset=['ra', 'dec']).reset_index(drop=True)
    print(f'\nCOSMOS table: {len(cosmos)} sources, RA {cosmos.ra.min():.3f}-{cosmos.ra.max():.3f}, '
          f'Dec {cosmos.dec.min():.3f}-{cosmos.dec.max():.3f}')
    pad = 0.01
    box = BOX or (cosmos.ra.min() - pad, cosmos.ra.max() + pad, cosmos.dec.min() - pad, cosmos.dec.max() + pad)
    desi = load_desi(box)
    print(f'DESI DR1 targets in the box: {len(desi)}')
    if desi.empty:
        raise SystemExit('No DESI DR1 targets in this box.')
    m = match(cosmos, desi, MATCH_ARCSEC)
    print(f'\nmatched within {MATCH_ARCSEC}": {len(m)} of {len(desi)} DESI targets')
    dup = m['cosmos_id'].duplicated(keep=False).sum()
    if dup:
        print(f'  {dup} rows share a COSMOS source with another DESI spectrum (kept; see the cosmos_id column)')
    if len(m):
        print(m.groupby(['spectype'])['targetid'].count().to_string())
    ok = (m['spectype'] == SPECTYPE) if SPECTYPE else pd.Series(True, index=m.index)
    if REQUIRE_ZWARN0:
        ok &= m['zwarn'] == 0
    if Z_RANGE is not None:
        ok &= m['z_desi'].between(*Z_RANGE)
    sel = m[ok].drop_duplicates('targetid')
    os.makedirs(OUT_DIR, exist_ok=True)
    m.to_csv(os.path.join(OUT_DIR, 'cosmos_desi_matches.csv'), index=False)
    sel[['targetid', 'survey', 'program', 'healpix', 'z_desi', 'cosmos_id', 'sep_arcsec']].to_csv(
        os.path.join(OUT_DIR, 'cosmos_desi_targetids.csv'), index=False)
    print(f'\n{len(sel)} targets pass the cuts (SPECTYPE={SPECTYPE}, ZWARN=0: {REQUIRE_ZWARN0}, Z_RANGE={Z_RANGE})')
    if 'dz' in m:
        d = m['dz'].dropna()
        if len(d):
            print(f'z_desi - z_phot: median {d.median():+.3f}, 16-84% [{d.quantile(.16):+.3f}, {d.quantile(.84):+.3f}]')
    print(f"\nwrote {os.path.join(OUT_DIR, 'cosmos_desi_matches.csv')} and cosmos_desi_targetids.csv"
          "\nnext: in unite_desi_dr1.py set  TARGETID_FILE = 'cosmos_desi_targetids.csv'")
    return m, sel


if __name__ == '__main__':
    m, sel = main()
