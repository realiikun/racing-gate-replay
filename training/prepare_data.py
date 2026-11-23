import math
def parse_label(path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        values = [float(x) for x in line.split()]
        if len(values) != 17 or not all(math.isfinite(x) for x in values):
            raise ValueError(f'{path}:{line_number}: expected 17 finite values')
        if values[0] != 0 or not all(0 <= x <= 1 for x in values[1:5]) or min(values[3:5]) <= 0:
            raise ValueError(f'{path}:{line_number}: invalid class or bbox')
        for i in range(5, 17, 3):
            x, y, v = values[i:i + 3]
            if v not in (0, 2):
                raise ValueError(f'{path}:{line_number}: unexpected visibility {v}')
            if v == 2 and not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f'{path}:{line_number}: visible corner outside image')
            if v == 0:
                # Invisible coordinates have no localization target. Normalize
                # only the derived training copy; original labels stay intact.
                values[i:i + 2] = [0., 0.]
        rows.append(values)
    return rows
