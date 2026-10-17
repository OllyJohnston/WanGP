#!/usr/bin/env bash
#
# Regenerate the WanGP Enhanced release assets.
#
# Produces two artefacts, both relative to the recorded upstream base commit
# (the first non-comment line of the UPSTREAM_BASE file at the repo root):
#
#     dist/wangp-enhanced-v<version>-vs-<base>.patch          unified diff
#     dist/wangp-enhanced-v<version>-vs-<base>-overlay.zip    changed files only
#
# Usage: scripts/make-patch.sh [--base <rev>] [--out-dir <dir>] [--repo <dir>]
#
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${script_dir}/.." && pwd)"
out_dir="dist"
base=""

while [ $# -gt 0 ]; do
    case "$1" in
        --base)    base="$2"; shift 2 ;;
        --out-dir) out_dir="$2"; shift 2 ;;
        --repo)    repo="$(cd "$2" && pwd)"; shift 2 ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$base" ]; then
    base_file="${repo}/UPSTREAM_BASE"
    [ -f "$base_file" ] || { echo "UPSTREAM_BASE not found at ${base_file}" >&2; exit 1; }
    base="$(grep -v '^[[:space:]]*#' "$base_file" | grep -v '^[[:space:]]*$' | head -n 1)"
fi
[ -n "$base" ] || { echo "could not determine upstream base commit" >&2; exit 1; }

version="$(sed -n 's/^[[:space:]]*WanGP_version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "${repo}/wgp.py" | head -n 1)"
[ -n "$version" ] || { echo "could not read WanGP_version from wgp.py" >&2; exit 1; }

short_base="$(git -C "$repo" rev-parse --short "$base")"
stem="wangp-enhanced-v${version}-vs-${short_base}"

case "$out_dir" in
    /*) : ;;
    *)  out_dir="${repo}/${out_dir}" ;;
esac
mkdir -p "$out_dir"

python_bin="${PYTHON:-}"
if [ -z "$python_bin" ]; then
    if command -v python3 >/dev/null 2>&1; then python_bin="python3"
    elif command -v python >/dev/null 2>&1; then python_bin="python"
    else echo "python is required to build the overlay zip" >&2; exit 1
    fi
fi

patch_path="${out_dir}/${stem}.patch"
zip_path="${out_dir}/${stem}-overlay.zip"

# Unified patch. bash redirection keeps git's LF output byte-for-byte - do not
# pipe through anything that re-encodes, or non-ASCII content (the README's
# emoji) is mangled and the patch stops applying.
git -C "$repo" --no-pager -c core.autocrlf=false diff --no-color --binary "$base" HEAD > "$patch_path"

# Overlay zip: only changed/added files, at their normal paths.
# (python -m zipfile would flatten them to basenames, so use the helper.)
mapfile -t files < <(git -C "$repo" --no-pager diff --name-only --diff-filter=ACMR "$base" HEAD)
[ "${#files[@]}" -gt 0 ] || { echo "no changed files relative to ${base}" >&2; exit 1; }
rm -f "$zip_path"
( cd "$repo" && "$python_bin" "${script_dir}/make_overlay_zip.py" "$zip_path" "${files[@]}" )

echo "base:    ${base} (${short_base})"
echo "files:   ${#files[@]}"
echo "patch:   ${patch_path}"
echo "overlay: ${zip_path}"
