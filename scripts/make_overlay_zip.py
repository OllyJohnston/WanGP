"""Create an overlay zip that preserves each file's repository-relative path.

``python -m zipfile -c`` stores every entry under its *basename*, which destroys
the directory structure an overlay archive needs (``shared/utils/model_scanner.py``
becomes ``model_scanner.py``). This helper writes each file at its full relative
path instead, so the archive can be extracted directly over a WanGP folder.

Usage:
    python scripts/make_overlay_zip.py <output.zip> <file> [<file> ...]

Files are resolved relative to the current working directory, so run this from
the repository root.
"""

from __future__ import annotations

import os
import sys
import zipfile


def build(zip_path, files, *, root="."):
    """Write ``files`` into ``zip_path`` at their relative paths.

    Returns the list of files actually written. Raises ``ValueError`` when no
    input file exists, so a silent empty archive is impossible.
    """
    present = [f for f in files if f and os.path.isfile(os.path.join(root, f))]
    if not present:
        raise ValueError("no input files exist")

    parent = os.path.dirname(os.path.abspath(zip_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in present:
            arcname = os.path.normpath(relative).replace(os.sep, "/")
            archive.write(os.path.join(root, relative), arcname)
    return present


def main(argv):
    if len(argv) < 3:
        print(__doc__.strip())
        return 2
    try:
        written = build(argv[1], argv[2:])
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"overlay: {argv[1]} ({len(written)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
