"""
core.py
========================
Drop-in replacement for the 2D interaction diagram section of core.py.
All measurements verified against pose1_interaction.svg (800 × 758.84 px,
Adobe Illustrator 30.2.1 export).

KEY FIXES vs previous version
──────────────────────────────
• Residue circle radius:   39.52 → 24.55  (exact SVG value)
• Residue label font-size: 23 px  → 14.29 px  (st9/st7/st16 classes)
• Placement radius R:      310   → 210        (measured from SVG centroid)
• Ligand target_size:      310   → 280        (ligand spans ~35 % of 800 px canvas)
• Canvas default size:   820×820 → 800×759    (exact SVG viewBox)
• Ligand centre Y:       H×0.44  → H×0.50    (ligand sits in true centre)
• Legend: replaces "all circles" with exact SVG legend:
    – hydrophobic → small blue circle r=9.54  (st4)
    – h-bond      → green dashed line  (st13)
    – π-π         → magenta dashed line (st3)
"""

import math as _math


# ══════════════════════════════════════════════════════════════════════════════
#  STYLE CONSTANTS  —  exact values from SVG class definitions
# ══════════════════════════════════════════════════════════════════════════════

_ITYPE_PRIORITY = [
    "metal", "ionic", "halogen", "hbond_to_halogen",
    "hbond", "pi_pi", "cation_pi", "hydrophobic",
]

# Exact SVG colours
_CLR_HBOND  = "#1a7a1a"   # st1/st2/st9/st13/st18
_CLR_PIPI   = "#e200e8"   # st3/st7/st11
_CLR_HYDRO  = "#2287ff"   # st4/st16
_CLR_IONIC  = "#aa0077"
_CLR_METAL  = "#cc8800"
_CLR_HAL    = "#cc2277"
_CLR_HBXHAL = "#6633aa"

# Residue background circles — fill + opacity (st4/st11/st15 etc.)
# Radius is 24.55 in every case (exact SVG measurement)
_RES_R = 24.55

_RES_CIRCLE = {
    "hbond":            dict(fill="#80dd80", opacity=0.2),   # st15
    "hbond_to_halogen": dict(fill="#80dd80", opacity=0.2),
    "pi_pi":            dict(fill="#e200e8", opacity=0.2),   # st11
    "cation_pi":        dict(fill="#e200e8", opacity=0.2),
    "hydrophobic":      dict(fill="#2287ff", opacity=0.2),   # st4
    "ionic":            dict(fill="#ffaae0", opacity=0.2),
    "metal":            dict(fill="#ffe080", opacity=0.2),
    "halogen":          dict(fill="#ffb0d0", opacity=0.2),
}

# Residue label colours — 14.29 px bold (st9 / st7 / st16)
_LBL_CLR = {
    "hbond":            _CLR_HBOND,   # st9
    "hbond_to_halogen": _CLR_HBOND,
    "pi_pi":            _CLR_PIPI,    # st7
    "cation_pi":        _CLR_PIPI,
    "hydrophobic":      _CLR_HYDRO,   # st16
    "ionic":            _CLR_IONIC,
    "metal":            _CLR_METAL,
    "halogen":          _CLR_HAL,
}

_LINE_CLR = {
    "hbond":            _CLR_HBOND,   # st13
    "hbond_to_halogen": _CLR_HBXHAL,
    "pi_pi":            _CLR_PIPI,    # st3
    "cation_pi":        _CLR_PIPI,
    "ionic":            _CLR_IONIC,
    "metal":            _CLR_METAL,
    "halogen":          _CLR_HAL,
    # hydrophobic: NO LINE — circle + label only (confirmed in SVG)
}

_ATOM_CLR = {
    "C": "#1a1a1a", "N": "#1a5fa8", "O": "#cc2222",
    "S": "#c8a800", "P": "#e07000", "F": "#1a7a1a",
    "CL": "#1a7a1a", "BR": "#8b2500", "I": "#5c2d8a", "H": "#555555",
}
_AROM_DOT_CLR = "#1a7a1a"   # st18

_METALS_SET     = {"MG","ZN","CA","MN","FE","CU","CO","NI","CD","HG","NA","K"}
_AROM_ATOMS     = {"PHE","TYR","TRP","HIS"}
_AROM_ATOM_NAMES = {
    "CG","CD1","CD2","CE1","CE2","CZ",
    "ND1","NE2","CE3","CZ2","CZ3","CH2",
}


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_aromatic_ring_data(mol, conf):
    import numpy as np
    results = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in (5, 6):
            continue
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        coords   = np.array([[conf.GetAtomPosition(i).x,
                               conf.GetAtomPosition(i).y,
                               conf.GetAtomPosition(i).z] for i in ring])
        centroid = coords.mean(axis=0)
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        n  = np.cross(v1, v2)
        nl = np.linalg.norm(n)
        if nl > 0:
            n /= nl
        results.append((centroid, n, list(ring)))
    return results


def _detect_all_interactions(lig_mol_3d, receptor_pdb: str,
                              cutoff: float = 4.5) -> list:
    import numpy as np
    from prody import parsePDB
    rec = parsePDB(receptor_pdb)
    if rec is None:
        return []
    rc  = rec.getCoords()
    rrn = rec.getResnames()
    rch = rec.getChids()
    rri = rec.getResnums()
    ran = rec.getNames()
    rel = rec.getElements()
    conf = lig_mol_3d.GetConformer()
    nl   = lig_mol_3d.GetNumAtoms()
    lxyz = np.array([[conf.GetAtomPosition(i).x,
                       conf.GetAtomPosition(i).y,
                       conf.GetAtomPosition(i).z] for i in range(nl)])
    latom = [lig_mol_3d.GetAtomWithIdx(i) for i in range(nl)]
    lel   = [a.GetSymbol().upper() for a in latom]
    larom = [a.GetIsAromatic() for a in latom]
    lchg  = [a.GetFormalCharge() for a in latom]

    POLAR = {"N","O","S","F"}
    HYDL  = {"C","S","CL","BR","I","F"}
    HYDR  = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","GLY","TYR","HIS"}

    results = []
    for j in range(len(rc)):
        rn  = rrn[j].strip(); ch = rch[j].strip(); ri = int(rri[j])
        el  = (rel[j].strip().upper() if rel[j] and rel[j].strip()
               else ran[j][:1].upper())
        rp  = rc[j]
        dists = np.linalg.norm(lxyz - rp, axis=1)
        md    = float(dists.min())
        mi    = int(dists.argmin())
        if md > max(cutoff + 1.0, 5.6):
            continue
        if el in POLAR:
            for i in range(nl):
                if lel[i] not in POLAR:
                    continue
                d = float(dists[i])
                if d < 3.5:
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                        itype="hbond", distance=round(d, 1), lig_atom_idx=i,
                        prot_el=el, is_donor=(el=="N"), ring_atom_indices=None))
                    break
        if el in {"C","S","CL","BR","I"} and rn in HYDR:
            for i in range(nl):
                if lel[i] not in HYDL:
                    continue
                d = float(dists[i])
                if d < cutoff:
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                        itype="hydrophobic", distance=round(d, 1), lig_atom_idx=i,
                        prot_el=el, is_donor=False, ring_atom_indices=None))
                    break
        if rn in {"ASP","GLU"} and el == "O":
            for i in range(nl):
                if lchg[i] > 0 and float(dists[i]) < 4.0:
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                        itype="ionic", distance=round(float(dists[i]),1),
                        lig_atom_idx=i, prot_el=el, is_donor=False,
                        ring_atom_indices=None))
                    break
        if rn in {"LYS","ARG"} and el == "N":
            for i in range(nl):
                if lchg[i] < 0 and float(dists[i]) < 4.0:
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                        itype="ionic", distance=round(float(dists[i]),1),
                        lig_atom_idx=i, prot_el=el, is_donor=True,
                        ring_atom_indices=None))
                    break
        if rn.strip().upper() in _METALS_SET or el in _METALS_SET:
            if md < 2.8:
                results.append(dict(resname=rn, chain=ch, resid=ri,
                    itype="metal", distance=round(md, 1), lig_atom_idx=mi,
                    prot_el=el, is_donor=False, ring_atom_indices=None))

    # π-π / cation-π: line must start from ring centroid
    lr = _get_aromatic_ring_data(lig_mol_3d, conf)
    if lr:
        for j in range(len(rc)):
            rn  = rrn[j].strip(); ch = rch[j].strip(); ri = int(rri[j])
            an  = ran[j].strip()
            if rn not in _AROM_ATOMS or an not in _AROM_ATOM_NAMES:
                continue
            rp = rc[j]
            for lc, _, ring_idxs in lr:
                d = float(np.linalg.norm(lc - rp))
                if d < 5.5:
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                        itype="pi_pi", distance=round(d, 1),
                        lig_atom_idx=ring_idxs[0],
                        prot_el="C", is_donor=False,
                        ring_atom_indices=ring_idxs))
                    break
        for j in range(len(rc)):
            rn  = rrn[j].strip(); ch = rch[j].strip(); ri = int(rri[j])
            el2 = (rel[j].strip().upper() if rel[j] and rel[j].strip()
                   else ran[j][:1].upper())
            if rn not in {"LYS","ARG"} or el2 != "N":
                continue
            rp = rc[j]
            for lc, _, ring_idxs in lr:
                d = float(np.linalg.norm(lc - rp))
                if d < 5.0:
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                        itype="cation_pi", distance=round(d, 1),
                        lig_atom_idx=ring_idxs[0], prot_el="N", is_donor=True,
                        ring_atom_indices=ring_idxs))
                    break

    # Halogen bonds
    _V  = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"S":1.80,"P":1.80,
           "F":1.47,"CL":1.75,"BR":1.85,"I":1.98}
    _XD = {17:"CL", 35:"BR", 53:"I"}
    _XA = {"O","N","S","P","F","CL","BR","I"}
    for i in range(nl):
        ano = latom[i].GetAtomicNum()
        if ano not in _XD:
            continue
        xel = _XD[ano]; xp = lxyz[i]; vx = _V.get(xel, 1.80)
        ri_ = next((nb.GetIdx() for nb in latom[i].GetNeighbors()
                    if nb.GetAtomicNum() == 6), None)
        if ri_ is None:
            continue
        rp2 = lxyz[ri_]
        for j in range(len(rc)):
            ael = (rel[j].strip().upper() if rel[j] and rel[j].strip()
                   else ran[j][:1].upper())
            isp = (rrn[j].strip() in _AROM_ATOMS
                   and ran[j].strip() in _AROM_ATOM_NAMES
                   and ael == "C")
            if ael not in _XA and not isp:
                continue
            ap = rc[j]
            d  = float(np.linalg.norm(xp - ap))
            if d > vx + _V.get(ael, 1.70):
                continue
            vRX = rp2 - xp; vXA = ap - xp
            import numpy as _np2
            ca  = _np2.dot(vRX, vXA) / (
                  _np2.linalg.norm(vRX) * _np2.linalg.norm(vXA) + 1e-9)
            if _math.degrees(_math.acos(max(-1.0, min(1.0, ca)))) >= 140:
                results.append(dict(resname=rrn[j].strip(), chain=rch[j].strip(),
                    resid=int(rri[j]), itype="halogen", distance=round(d,1),
                    lig_atom_idx=i, prot_el=ael, is_donor=False,
                    ring_atom_indices=None))

    # H-bond to halogen
    _HA = {9:"F", 17:"CL", 35:"BR", 53:"I"}
    _HD = {"O","N","S"}
    for i in range(nl):
        ano = latom[i].GetAtomicNum()
        if ano not in _HA:
            continue
        xel = _HA[ano]; xp = lxyz[i]; vx = _V.get(xel, 1.80)
        ri2 = next((nb.GetIdx() for nb in latom[i].GetNeighbors()), None)
        if ri2 is None:
            continue
        rlp = lxyz[ri2]
        for j in range(len(rc)):
            hel = (rel[j].strip().upper() if rel[j] and rel[j].strip()
                   else ran[j][:1].upper())
            if hel != "H":
                continue
            hp  = rc[j]
            dhx = float(np.linalg.norm(hp - xp))
            if dhx > _V["H"] + vx:
                continue
            dp = None
            for k in range(len(rc)):
                if k == j:
                    continue
                dk = (rel[k].strip().upper() if rel[k] and rel[k].strip()
                      else ran[k][:1].upper())
                if dk not in _HD:
                    continue
                if float(np.linalg.norm(rc[k] - hp)) < 1.15:
                    dp = rc[k]; break
            if dp is None:
                continue
            vHD = dp - hp; vHX = xp - hp
            import numpy as _np3
            cd = _np3.dot(vHD, vHX) / (
                 _np3.linalg.norm(vHD) * _np3.linalg.norm(vHX) + 1e-9)
            if _math.degrees(_math.acos(max(-1.0, min(1.0, cd)))) < 120:
                continue
            vXR = rlp - xp; vXH = hp - xp
            cr2 = _np3.dot(vXR, vXH) / (
                  _np3.linalg.norm(vXR) * _np3.linalg.norm(vXH) + 1e-9)
            arx = _math.degrees(_math.acos(max(-1.0, min(1.0, cr2))))
            if not (70 <= arx <= 120):
                continue
            results.append(dict(resname=rrn[j].strip(), chain=rch[j].strip(),
                resid=int(rri[j]), itype="hbond_to_halogen", distance=round(dhx,1),
                lig_atom_idx=i, prot_el="N", is_donor=True, ring_atom_indices=None))
    return results


def _deduplicate_interactions(interactions: list) -> list:
    priority = {t: i for i, t in enumerate(_ITYPE_PRIORITY)}
    best: dict = {}
    for ix in interactions:
        key = (ix["chain"], ix["resid"])
        if key not in best:
            best[key] = ix
        else:
            pn = priority.get(ix["itype"], 99)
            po = priority.get(best[key]["itype"], 99)
            if pn < po or (pn == po and ix["distance"] < best[key]["distance"]):
                best[key] = ix
    return list(best.values())


def _compute_svg_coords(mol2d, cx, cy, target_size=280):
    from rdkit.Chem import rdDepictor
    if mol2d.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol2d)
    conf = mol2d.GetConformer()
    n    = mol2d.GetNumAtoms()
    if n == 0:
        return {}
    xs = [conf.GetAtomPosition(i).x for i in range(n)]
    ys = [conf.GetAtomPosition(i).y for i in range(n)]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.01)
    sc   = target_size / span
    mx   = (min(xs) + max(xs)) / 2
    my   = (min(ys) + max(ys)) / 2
    return {i: (cx + (xs[i] - mx) * sc, cy - (ys[i] - my) * sc)
            for i in range(n)}


def _ring_centroid_2d(ring_atom_indices, svg_coords):
    xs = [svg_coords[i][0] for i in ring_atom_indices if i in svg_coords]
    ys = [svg_coords[i][1] for i in ring_atom_indices if i in svg_coords]
    if not xs:
        return None, None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _place_residues_no_cross(interactions, svg_coords, cx, cy, R=210):
    """Angular-sort placement so lines never cross."""
    if not interactions:
        return []
    anchored = []
    for ix in interactions:
        if ix.get("ring_atom_indices"):
            ax, ay = _ring_centroid_2d(ix["ring_atom_indices"], svg_coords)
            if ax is None:
                ax, ay = svg_coords.get(ix.get("lig_atom_idx", 0), (cx, cy))
        else:
            ai = ix.get("lig_atom_idx", 0)
            ax, ay = svg_coords.get(ai, (cx, cy))
        anchored.append({**ix, "anchor_angle": _math.atan2(ay - cy, ax - cx)})
    anchored.sort(key=lambda x: x["anchor_angle"])
    n     = len(anchored)
    slots = [-_math.pi / 2 + (2 * _math.pi * k / n) for k in range(n)]
    used  = [False] * n
    result = []
    for res in anchored:
        aa = res["anchor_angle"]
        bd, bs = float("inf"), 0
        for s in range(n):
            if used[s]:
                continue
            d = abs(_math.atan2(_math.sin(slots[s] - aa),
                                 _math.cos(slots[s] - aa)))
            if d < bd:
                bd, bs = d, s
        used[bs] = True
        result.append({**res,
            "bx": cx + R * _math.cos(slots[bs]),
            "by": cy + R * _math.sin(slots[bs]),
            "slot_angle": slots[bs]})
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  LIGAND SVG RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _render_ligand_svg(mol2d, svg_coords):
    from rdkit import Chem
    parts = []
    ri       = mol2d.GetRingInfo()
    arom_bonds = set()
    arom_rings = []
    for ring in ri.AtomRings():
        if all(mol2d.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            for k in range(len(ring)):
                arom_bonds.add(frozenset([ring[k], ring[(k + 1) % len(ring)]]))
            arom_rings.append(ring)

    def _sh(fx, fy, tx, ty, sym):
        """Shorten bond endpoint so it doesn't overdraw the atom label."""
        if sym not in ("C", ""):
            dx, dy = tx - fx, ty - fy
            L = _math.sqrt(dx * dx + dy * dy) + 1e-9
            r = {"H":8,"N":9,"O":9,"S":11,"P":11,"F":8,"CL":13,"BR":13,"I":11}.get(sym, 9)
            return fx + dx / L * r, fy + dy / L * r
        return fx, fy

    for bond in mol2d.GetBonds():
        i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        x1, y1 = svg_coords.get(i1, (0, 0))
        x2, y2 = svg_coords.get(i2, (0, 0))
        s1 = mol2d.GetAtomWithIdx(i1).GetSymbol().upper()
        s2 = mol2d.GetAtomWithIdx(i2).GetSymbol().upper()
        x1s, y1s = _sh(x1, y1, x2, y2, s1)
        x2s, y2s = _sh(x2, y2, x1, y1, s2)
        bt = bond.GetBondType()

        if frozenset([i1, i2]) in arom_bonds:
            # st2: stroke-width 1.77, opacity 0.9
            parts.append(
                f'<line x1="{x1s:.2f}" y1="{y1s:.2f}"'
                f' x2="{x2s:.2f}" y2="{y2s:.2f}"'
                f' stroke="#1a1a1a" stroke-width="1.77" opacity="0.9"/>')
        elif bt == Chem.BondType.DOUBLE:
            # st10: stroke-width 2.44, opacity 0.9
            dx, dy = x2s - x1s, y2s - y1s
            L  = _math.sqrt(dx * dx + dy * dy) + 1e-9
            px, py = -dy / L * 3.8, dx / L * 3.8
            for sg in (1, -1):
                parts.append(
                    f'<line x1="{x1s+px*sg:.2f}" y1="{y1s+py*sg:.2f}"'
                    f' x2="{x2s+px*sg:.2f}" y2="{y2s+py*sg:.2f}"'
                    f' stroke="#1a1a1a" stroke-width="2.44" opacity="0.9"/>')
        elif bt == Chem.BondType.TRIPLE:
            dx, dy = x2s - x1s, y2s - y1s
            L  = _math.sqrt(dx * dx + dy * dy) + 1e-9
            px, py = -dy / L * 4.5, dx / L * 4.5
            for m in (-1, 0, 1):
                parts.append(
                    f'<line x1="{x1s+px*m:.2f}" y1="{y1s+py*m:.2f}"'
                    f' x2="{x2s+px*m:.2f}" y2="{y2s+py*m:.2f}"'
                    f' stroke="#1a1a1a" stroke-width="2.0" opacity="0.9"/>')
        else:
            bd = bond.GetBondDir()
            if bd == Chem.BondDir.BEGINWEDGE:
                dx, dy = x2s - x1s, y2s - y1s
                L  = _math.sqrt(dx * dx + dy * dy) + 1e-9
                px, py = -dy / L * 5.0, dx / L * 5.0
                parts.append(
                    f'<polygon points="{x1s:.2f},{y1s:.2f}'
                    f' {x2s+px:.2f},{y2s+py:.2f} {x2s-px:.2f},{y2s-py:.2f}"'
                    f' fill="#1a1a1a" stroke="none"/>')
            elif bd == Chem.BondDir.BEGINDASH:
                dx, dy = x2s - x1s, y2s - y1s
                L  = _math.sqrt(dx * dx + dy * dy) + 1e-9
                px, py = -dy / L, dx / L
                for step in range(1, 6):
                    t   = step / 7
                    mx2 = x1s + dx * t; my2 = y1s + dy * t
                    w   = t * 5.0
                    parts.append(
                        f'<line x1="{mx2-px*w:.2f}" y1="{my2-py*w:.2f}"'
                        f' x2="{mx2+px*w:.2f}" y2="{my2+py*w:.2f}"'
                        f' stroke="#1a1a1a" stroke-width="1.8"/>')
            else:
                # st2 single bond
                parts.append(
                    f'<line x1="{x1s:.2f}" y1="{y1s:.2f}"'
                    f' x2="{x2s:.2f}" y2="{y2s:.2f}"'
                    f' stroke="#1a1a1a" stroke-width="1.77" opacity="0.9"/>')

    # Aromatic rings: dashed circle (st14) + green centroid dot (st18)
    for ring in arom_rings:
        rcoords = [svg_coords.get(i, (0, 0)) for i in ring]
        rcx = sum(x for x, y in rcoords) / len(rcoords)
        rcy = sum(y for x, y in rcoords) / len(rcoords)
        avg = sum(_math.sqrt((x - rcx) ** 2 + (y - rcy) ** 2)
                  for x, y in rcoords) / len(rcoords)
        cr  = avg * 0.58
        # st14: fill none, stroke #1a1a1a, stroke-width 1.77, opacity 0.7,
        #       stroke-dasharray "5.43 2.72"
        parts.append(
            f'<circle cx="{rcx:.2f}" cy="{rcy:.2f}" r="{cr:.2f}"'
            f' fill="none" stroke="#1a1a1a" stroke-width="1.77"'
            f' stroke-dasharray="5.43 2.72" opacity="0.7"/>')
        # st18: fill #1a7a1a, r=5.43
        parts.append(
            f'<circle cx="{rcx:.2f}" cy="{rcy:.2f}" r="5.43"'
            f' fill="{_AROM_DOT_CLR}"/>')

    # Heteroatom labels: white knock-out rect + coloured bold text
    # st0 (N): fill #1a5fa8, 17.65 px bold
    # st12 (O): fill #cc2222, 17.65 px bold
    for i in range(mol2d.GetNumAtoms()):
        atom = mol2d.GetAtomWithIdx(i)
        sym  = atom.GetSymbol()
        if sym == "C":
            continue
        ax, ay = svg_coords.get(i, (0, 0))
        clr = _ATOM_CLR.get(sym.upper(), "#555")
        fs  = {"H": 16}.get(sym, 17.65)
        hw  = {"H":7,"N":9,"O":9,"S":11,"P":11,"F":8,"CL":16,"BR":16,"I":11
               }.get(sym.upper(), 9)
        parts.append(
            f'<rect x="{ax-hw:.1f}" y="{ay-11:.1f}"'
            f' width="{hw*2:.0f}" height="22" fill="white" stroke="none"/>')
        parts.append(
            f'<text x="{ax:.2f}" y="{ay:.2f}" text-anchor="middle"'
            f' dominant-baseline="central"'
            f' font-family="Arial,sans-serif" font-size="{fs}"'
            f' font-weight="700" fill="{clr}">{sym}</text>')
        fc = atom.GetFormalCharge()
        if fc != 0:
            fcs = "+" if fc == 1 else "−" if fc == -1 else f"{fc:+d}"
            parts.append(
                f'<text x="{ax+hw:.1f}" y="{ay-hw+2:.1f}"'
                f' font-family="Arial,sans-serif" font-size="10"'
                f' fill="{clr}">{fcs}</text>')
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SVG RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _render_diagram_svg(mol2d, svg_coords, placements, title, W, H):
    parts = []
    parts.append(
        f'<svg width="100%" viewBox="0 0 {W} {H}"'
        f' xmlns="http://www.w3.org/2000/svg">')
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    # ── Title pill  (st19 fill #f2f2f2, st5 font 24.93 px bold) ──────────────
    if title:
        esc = (title.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
        tw  = len(esc) * 14.5 + 48
        tw  = max(tw, 240)
        tw  = min(tw, W - 40)
        px  = (W - tw) / 2
        ph  = 46
        pr  = ph / 2
        parts.append(
            f'<rect x="{px:.1f}" y="14" width="{tw:.0f}" height="{ph}"'
            f' rx="{pr:.1f}" ry="{pr:.1f}" fill="#f2f2f2" stroke="none"/>')
        parts.append(
            f'<text x="{W/2:.1f}" y="37" text-anchor="middle"'
            f' dominant-baseline="central"'
            f' font-family="Arial,sans-serif" font-size="24.93"'
            f' font-weight="700" fill="#1a1a1a">{esc}</text>')

    # ── PASS 1: Residue background circles  (drawn first — behind lines) ──────
    # Radius: 24.55  (exact SVG measurement — was incorrectly 39.52)
    for p in placements:
        itype = p["itype"]
        bx, by = p["bx"], p["by"]
        cbx = max(50, min(bx, W - 50))
        cby = max(70, min(by, H - 65))
        bg  = _RES_CIRCLE.get(itype, dict(fill="#cccccc", opacity=0.2))
        parts.append(
            f'<circle cx="{cbx:.1f}" cy="{cby:.1f}" r="{_RES_R}"'
            f' fill="{bg["fill"]}" opacity="{bg["opacity"]}"/>')

    # ── PASS 2: Interaction lines ─────────────────────────────────────────────
    for p in placements:
        itype = p["itype"]
        if itype == "hydrophobic":
            continue   # NO LINE — circle + label only (confirmed in SVG)

        bx, by = p["bx"], p["by"]
        cbx = max(50, min(bx, W - 50))
        cby = max(70, min(by, H - 65))

        # Line origin: ring centroid for π-π, else ligand atom position
        if itype in ("pi_pi", "cation_pi") and p.get("ring_atom_indices"):
            lx, ly = _ring_centroid_2d(p["ring_atom_indices"], svg_coords)
            if lx is None:
                ai = p.get("lig_atom_idx", 0)
                lx, ly = svg_coords.get(ai, (W // 2, H // 2))
        else:
            ai = p.get("lig_atom_idx", 0)
            lx, ly = svg_coords.get(ai, (W // 2, H // 2))

        clr = _LINE_CLR.get(itype, "#888")

        if itype in ("pi_pi", "cation_pi"):
            # st3: stroke #e200e8, dasharray "5 3", stroke-width 1.6, opacity 0.85
            parts.append(
                f'<line x1="{lx:.2f}" y1="{ly:.2f}"'
                f' x2="{cbx:.2f}" y2="{cby:.2f}"'
                f' stroke="{clr}" stroke-width="1.6"'
                f' stroke-dasharray="5 3" opacity="0.85"/>')

        elif itype in ("hbond", "hbond_to_halogen"):
            # st13: stroke #1a7a1a, dasharray "5 3", stroke-width 1.6, opacity 0.85
            da = ('stroke-dasharray="5 3"' if itype == "hbond"
                  else 'stroke-dasharray="4 2 1 2"')
            parts.append(
                f'<line x1="{lx:.2f}" y1="{ly:.2f}"'
                f' x2="{cbx:.2f}" y2="{cby:.2f}"'
                f' stroke="{clr}" stroke-width="1.6"'
                f' {da} opacity="0.85"/>')
            # Distance label on line: st1 — 14 px bold, fill #1a7a1a
            if p.get("distance") is not None:
                mx2 = (lx + cbx) / 2; my2 = (ly + cby) / 2
                ds  = f"{p['distance']}\u00c5"
                tw2 = len(ds) * 7 + 8
                parts.append(
                    f'<rect x="{mx2-tw2/2:.1f}" y="{my2-9:.1f}"'
                    f' width="{tw2:.0f}" height="17" rx="4"'
                    f' fill="white" stroke="{clr}" stroke-width="0.5"/>')
                parts.append(
                    f'<text x="{mx2:.1f}" y="{my2:.1f}" text-anchor="middle"'
                    f' dominant-baseline="central"'
                    f' font-family="Arial,sans-serif" font-size="14"'
                    f' font-weight="700" fill="{clr}">{ds}</text>')

        else:
            # ionic / metal / halogen
            da = {"ionic":"6 2 2 2","metal":"3 2","halogen":"5 2"}.get(itype,"5 3")
            parts.append(
                f'<line x1="{lx:.2f}" y1="{ly:.2f}"'
                f' x2="{cbx:.2f}" y2="{cby:.2f}"'
                f' stroke="{clr}" stroke-width="1.8"'
                f' stroke-dasharray="{da}" opacity="0.85"/>')
            if p.get("distance") is not None:
                mx2 = (lx + cbx) / 2; my2 = (ly + cby) / 2
                ds  = f"{p['distance']}\u00c5"
                tw2 = len(ds) * 7 + 8
                parts.append(
                    f'<rect x="{mx2-tw2/2:.1f}" y="{my2-9:.1f}"'
                    f' width="{tw2:.0f}" height="17" rx="4"'
                    f' fill="white" stroke="{clr}" stroke-width="0.5"/>')
                parts.append(
                    f'<text x="{mx2:.1f}" y="{my2:.1f}" text-anchor="middle"'
                    f' dominant-baseline="central"'
                    f' font-family="Arial,sans-serif" font-size="14"'
                    f' font-weight="700" fill="{clr}">{ds}</text>')

    # ── PASS 3: Ligand structure ──────────────────────────────────────────────
    parts.append(_render_ligand_svg(mol2d, svg_coords))

    # ── PASS 4: Residue labels ────────────────────────────────────────────────
    # st9/st7/st16: 14.29 px bold (was 23 px — corrected to match SVG)
    for p in placements:
        itype = p["itype"]
        bx, by = p["bx"], p["by"]
        cbx  = max(50, min(bx, W - 50))
        cby  = max(70, min(by, H - 65))
        rn   = p["resname"]
        ri   = p["resid"]
        ch   = p.get("chain", "")
        lbl  = f"{rn.upper()} {ri}{ch}"
        lclr = _LBL_CLR.get(itype, "#333")
        parts.append(
            f'<text x="{cbx:.1f}" y="{cby:.1f}" text-anchor="middle"'
            f' dominant-baseline="central"'
            f' font-family="Arial,sans-serif" font-size="14.29"'
            f' font-weight="700" fill="{lclr}">{lbl}</text>')

    # ── PASS 5: Legend ────────────────────────────────────────────────────────
    # Exact SVG legend layout (bottom strip, y ≈ H-150 to H-108):
    #   hydrophobic → small blue circle (st4, r=9.54)
    #   h-bond      → short green dashed line (st13)
    #   π-π         → short magenta dashed line (st3)
    #   (other types if present: circle in their colour)
    #
    # Legend outer rect: st17 — stroke #f2f2f2, fill none
    _LEG_ORDER = [
        "hydrophobic", "hbond", "pi_pi", "cation_pi",
        "hbond_to_halogen", "ionic", "metal", "halogen",
    ]
    _LEG_LABEL = {
        "hydrophobic":      "Hydrophobic",
        "hbond":            "Hydrogen bond",
        "hbond_to_halogen": "H···Halogen",
        "pi_pi":            "π-π stacking",
        "cation_pi":        "Cation-π",
        "ionic":            "Ionic",
        "metal":            "Metal",
        "halogen":          "Halogen bond",
    }
    # Line-style legend glyphs (match SVG reference exactly)
    _LEG_LINE_CLR = {
        "hbond":            _CLR_HBOND,
        "hbond_to_halogen": _CLR_HBXHAL,
        "pi_pi":            _CLR_PIPI,
        "cation_pi":        _CLR_PIPI,
        "ionic":            _CLR_IONIC,
        "metal":            _CLR_METAL,
        "halogen":          _CLR_HAL,
    }
    _LEG_LINE_DASH = {
        "hbond":            "5 3",
        "hbond_to_halogen": "4 2 1 2",
        "pi_pi":            "5 3",
        "cation_pi":        "5 3",
        "ionic":            "6 2 2 2",
        "metal":            "3 2",
        "halogen":          "5 2",
    }

    active = [t for t in _LEG_ORDER if any(p["itype"] == t for p in placements)]
    if active:
        # Measure each item: glyph (32 px) + label text + padding
        item_widths = [len(_LEG_LABEL.get(t, t)) * 9.5 + 52 for t in active]
        total_w = sum(item_widths) + 16
        total_w = min(total_w, W - 20)
        lx0     = (W - total_w) / 2
        ly0     = H - 50   # vertical centre of legend strip

        # Legend border rect (st17: stroke #f2f2f2, fill none)
        parts.append(
            f'<rect x="{lx0:.0f}" y="{ly0-18}" width="{total_w:.0f}"'
            f' height="36" fill="none" stroke="#f2f2f2"'
            f' stroke-miterlimit="10"/>')

        cur_x = lx0 + 8
        for it in active:
            lbl_txt = _LEG_LABEL.get(it, it)
            lbl_w   = len(lbl_txt) * 9.5 + 44

            if it == "hydrophobic":
                # Small blue circle (st4, r=9.54)
                bg = _RES_CIRCLE.get(it, dict(fill="#2287ff", opacity=0.2))
                parts.append(
                    f'<circle cx="{cur_x+9:.1f}" cy="{ly0:.1f}" r="9.54"'
                    f' fill="{bg["fill"]}" opacity="{bg["opacity"]}"/>')
                parts.append(
                    f'<text x="{cur_x+24:.1f}" y="{ly0:.1f}"'
                    f' text-anchor="start" dominant-baseline="central"'
                    f' font-family="Arial,sans-serif" font-size="16"'
                    f' font-weight="700" fill="#555">{lbl_txt}</text>')
            else:
                # Dashed line glyph (32 px wide)
                lclr = _LEG_LINE_CLR.get(it, "#888")
                da   = _LEG_LINE_DASH.get(it, "5 3")
                sw   = "1.6" if it in ("hbond","hbond_to_halogen","pi_pi","cation_pi") else "1.8"
                parts.append(
                    f'<line x1="{cur_x:.1f}" y1="{ly0:.1f}"'
                    f' x2="{cur_x+32:.1f}" y2="{ly0:.1f}"'
                    f' stroke="{lclr}" stroke-width="{sw}"'
                    f' stroke-dasharray="{da}" opacity="0.85"/>')
                parts.append(
                    f'<text x="{cur_x+38:.1f}" y="{ly0:.1f}"'
                    f' text-anchor="start" dominant-baseline="central"'
                    f' font-family="Arial,sans-serif" font-size="16"'
                    f' font-weight="700" fill="#555">{lbl_txt}</text>')

            cur_x += lbl_w

    parts.append('</svg>')
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def draw_interaction_diagram(
    receptor_pdb: str,
    pose_sdf: str,
    smiles: str,
    title: str = "",
    cutoff: float = 4.5,
    size: tuple = (800, 759),   # exact SVG viewBox dimensions
    max_residues: int = 14,
) -> bytes:
    """
    2D interaction diagram — style matched to pose1_interaction.svg.

    Measurements verified against reference:
      • Canvas:          800 × 759
      • Residue circles: r = 24.55, opacity = 0.2
      • Residue labels:  14.29 px bold (coloured per interaction type)
      • Placement R:     210 px from ligand centre
      • Ligand size:     target_size = 280 (≈ 35 % of canvas width)
      • Hydrophobic:     circle + label ONLY — no line
      • H-bond:          green dashed line (st13) + circle + distance label
      • π-π / cation-π:  magenta dashed line (st3) from ring centroid + circle
      • Legend:          dashed lines for h-bond/π-π,
                         small circle (r=9.54) for hydrophobic  ← exact SVG
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor
    RDLogger.DisableLog("rdApp.*")
    W, H = size

    # ── Load 3-D pose ─────────────────────────────────────────────────────────
    try:
        mol3d = None
        for san in (True, False):
            sup   = Chem.SDMolSupplier(pose_sdf, sanitize=san, removeHs=False)
            mol3d = next((m for m in sup if m is not None), None)
            if mol3d is not None:
                if not san:
                    try: Chem.SanitizeMol(mol3d)
                    except: pass
                break
        if mol3d is None or mol3d.GetNumConformers() == 0:
            raise ValueError("No valid 3D pose in SDF")
    except Exception as e:
        RDLogger.EnableLog("rdApp.error")
        return (f'<svg viewBox="0 0 {W} 80" xmlns="http://www.w3.org/2000/svg">'
                f'<rect width="{W}" height="80" fill="white"/>'
                f'<text x="{W//2}" y="44" text-anchor="middle"'
                f' font-family="Arial,sans-serif" font-size="13" fill="#cc2222">'
                f'Error: {e}</text></svg>').encode()

    # ── Build 2-D depiction molecule ──────────────────────────────────────────
    mol2d = None
    if smiles and smiles.strip():
        mol2d = Chem.MolFromSmiles(smiles.strip())
    if mol2d is None:
        mol2d = Chem.RemoveHs(mol3d, sanitize=False)
        try: Chem.SanitizeMol(mol2d)
        except: pass
    mol2d = Chem.RemoveHs(mol2d)
    rdDepictor.Compute2DCoords(mol2d)

    # Map 3-D atom indices → 2-D atom indices
    m3 = Chem.RemoveHs(mol3d, sanitize=False)
    try: Chem.SanitizeMol(m3)
    except: pass
    m3to2d: dict = {}
    try:
        mt = m3.GetSubstructMatch(mol2d)
        if len(mt) == mol2d.GetNumAtoms():
            for i2, i3 in enumerate(mt):
                m3to2d[i3] = i2
    except: pass

    # ── Detect interactions ───────────────────────────────────────────────────
    try:
        raw = _detect_all_interactions(mol3d, receptor_pdb, cutoff=cutoff)
    except:
        raw = []
    for ix in raw:
        ix["lig_atom_idx"] = m3to2d.get(ix.get("lig_atom_idx", 0), 0)
        if ix.get("ring_atom_indices"):
            ix["ring_atom_indices"] = [
                m3to2d.get(i, i) for i in ix["ring_atom_indices"]
            ]

    pm  = {t: i for i, t in enumerate(_ITYPE_PRIORITY)}
    ded = _deduplicate_interactions(raw)
    ded.sort(key=lambda x: (pm.get(x["itype"], 99), x["distance"]))
    ded = ded[:max_residues]

    # ── Compute 2-D layout ────────────────────────────────────────────────────
    # Ligand centred at (W/2, H/2) — true centre, matching SVG proportions.
    # target_size=280 gives a ligand ~35 % of canvas width (verified from SVG).
    # R=210: measured from SVG centroid to outermost residue circles.
    cx = W // 2
    cy = H // 2
    sc = _compute_svg_coords(mol2d, cx, cy, target_size=280)
    pl = _place_residues_no_cross(ded, sc, cx, cy, R=210)

    svg = _render_diagram_svg(mol2d, sc, pl, title, W, H)
    RDLogger.EnableLog("rdApp.error")
    return svg.encode()


def draw_interactions_rdkit(lig_mol, receptor_pdb: str, smiles: str,
                             title: str = "", cutoff: float = 3.5,
                             size: tuple = (500, 500),
                             max_residues: int = 10) -> bytes:
    """Backward-compatible alias → draw_interaction_diagram."""
    import tempfile
    from rdkit import Chem
    tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    with Chem.SDWriter(tmp.name) as w:
        w.write(lig_mol)
    return draw_interaction_diagram(
        receptor_pdb=receptor_pdb,
        pose_sdf=tmp.name,
        smiles=smiles,
        title=title,
        cutoff=cutoff,
        size=(800, 759),
        max_residues=max_residues,
    )


def _svg_stamp(svg_text: str, title: str, w: int, h: int) -> str:
    esc = (title.replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;"))
    pad  = int(w * 0.05)
    pw   = w - 2 * pad
    ph   = 28
    py   = h - ph - 8
    ty   = py + ph // 2
    r    = ph // 2
    st   = (
        f'<g><rect x="{pad}" y="{py}" width="{pw}" height="{ph}"'
        f' rx="{r}" ry="{r}"'
        f' fill="#E8E8E8" fill-opacity="0.93"'
        f' stroke="#C8C8C8" stroke-width="0.5"/>'
        f'<text x="{w//2}" y="{ty}" text-anchor="middle"'
        f' dominant-baseline="middle"'
        f' font-family="Arial,sans-serif" font-size="13"'
        f' font-weight="500" fill="#1A1A1A">{esc}</text></g>'
    )
    return svg_text.replace("</svg>", f"{st}</svg>")
