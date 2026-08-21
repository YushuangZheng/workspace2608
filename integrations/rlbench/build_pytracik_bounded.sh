#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /absolute/path/to/conda /absolute/empty/install-prefix" >&2
  exit 2
fi

conda_bin=$(realpath "$1")
install_prefix=$(realpath -m "$2")
if [[ ! -x "$conda_bin" ]]; then
  echo "Conda executable is unavailable: $conda_bin" >&2
  exit 2
fi
if [[ "$install_prefix" == "/" || "$install_prefix" == "$HOME" ]]; then
  echo "Refusing broad install prefix: $install_prefix" >&2
  exit 2
fi
if [[ -e "$install_prefix" ]] && [[ -n "$(find "$install_prefix" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Install prefix must be absent or empty: $install_prefix" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
patch_path="$script_dir/patches/pytracik-0.0.3-bounded-cartesian-api.patch"
lock_path="$script_dir/third_party/pytracik/conda-linux-64-cp38.lock"
expected_patch_sha=ce1f9053bfc70b6d1a562fe2c830a8f04d8cc59a985941fc3163c3268fe4b565
actual_patch_sha=$(sha256sum "$patch_path" | awk '{print $1}')
[[ "$actual_patch_sha" == "$expected_patch_sha" ]] || {
  echo "Bounded API patch digest mismatch" >&2
  exit 1
}
echo "22b236e3cff3df2fdaf736390ed0583b93c8cbe2adaee9d0c64d4e5e5e4ba7d7  $lock_path" | sha256sum -c -

# Build and runtime libraries live in one exact, isolated conda prefix.  The
# explicit lock includes package build strings and MD5s from the validated
# CPython 3.8 environment.
"$conda_bin" create --yes --prefix "$install_prefix" --file "$lock_path"
python_bin="$install_prefix/bin/python"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 8), sys.version'

work_root=$(mktemp -d /tmp/dynamac-pytracik-build.XXXXXX)
cleanup() { rm -rf -- "$work_root"; }
trap cleanup EXIT
archive="$work_root/source.tar.gz"
curl -fsSL --retry 3 \
  https://github.com/chenhaox/pytracik/archive/8c8fd2d8ca70334af9b747987f1156ebb1da25cc.tar.gz \
  -o "$archive"
echo "6ab16912c8bbad74214b553a6b49ac39f3a48362b512f187fc08d852af946f8a  $archive" | sha256sum -c -
tar -xzf "$archive" -C "$work_root"
source_root=$(find "$work_root" -mindepth 1 -maxdepth 1 -type d -name 'pytracik-*' -print -quit)
[[ -n "$source_root" ]] || { echo "Pinned source archive is malformed" >&2; exit 1; }
patch --directory="$source_root" --strip=1 --forward --input="$patch_path"

export CPATH="$install_prefix/include:$install_prefix/include/eigen3:$install_prefix/include/orocos"
export LIBRARY_PATH="$install_prefix/lib"
export LD_LIBRARY_PATH="$install_prefix/lib"
export LDFLAGS="-L$install_prefix/lib -Wl,-rpath,$install_prefix/lib"
"$python_bin" -m pip install --disable-pip-version-check \
  --no-build-isolation --no-deps "$source_root"

purelib=$("$python_bin" - <<'PY'
import sysconfig
print(sysconfig.get_path("purelib"))
PY
)
PYTHONPATH="$purelib" "$python_bin" - "$purelib" <<'PY'
import hashlib, json, pathlib, sys, sysconfig
purelib = pathlib.Path(sys.argv[1]).resolve()
import pytracik
from trac_ik import TracIK
extension = pathlib.Path(pytracik.__file__).resolve()
suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
assert extension.name.endswith(suffix), (extension, suffix)
assert callable(getattr(pytracik, "ik_with_bounds", None))
assert callable(getattr(TracIK, "ik_with_bounds", None))
digest = hashlib.sha256(extension.read_bytes()).hexdigest()
manifest = {
    "build_id": "pytracik-v0.0.3-8c8fd2d8-bounded-cartesian-ce1f9053bfc7-cpython38-v1",
    "package_version": "0.0.3",
    "upstream_commit": "8c8fd2d8ca70334af9b747987f1156ebb1da25cc",
    "source_archive_sha256": "6ab16912c8bbad74214b553a6b49ac39f3a48362b512f187fc08d852af946f8a",
    "bounded_patch_sha256": "ce1f9053bfc70b6d1a562fe2c830a8f04d8cc59a985941fc3163c3268fe4b565",
    "conda_lock_sha256": "22b236e3cff3df2fdaf736390ed0583b93c8cbe2adaee9d0c64d4e5e5e4ba7d7",
    "python_abi": "cp38",
    "native_extension_sha256": digest,
}
(purelib / "trac_ik" / "_dynamac_build.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

# Expose only pytracik itself to the existing simulator interpreter.  Adding
# the complete CPython-3.8 conda site-packages directory to PYTHONPATH would
# also leak its NumPy into the separate CPython-3.10 policy worker.
overlay="$install_prefix/formal-overlay"
mkdir -p "$overlay"
extension=$(find "$purelib" -maxdepth 1 -type f -name 'pytracik*.so' -print -quit)
dist_info=$(find "$purelib" -maxdepth 1 -type d -name 'pytracik-*.dist-info' -print -quit)
[[ -n "$extension" && -n "$dist_info" ]] || {
  echo "Installed pytracik artifacts are incomplete" >&2
  exit 1
}
ln -s "$extension" "$overlay/$(basename "$extension")"
ln -s "$purelib/trac_ik" "$overlay/trac_ik"
ln -s "$dist_info" "$overlay/$(basename "$dist_info")"

echo "Installed pinned bounded pytracik under: $install_prefix"
echo "Minimal overlay: $overlay"
echo "Keep using the existing PyRep/RLBench simulator interpreter."
echo "Prepend $overlay to PYTHONPATH and $install_prefix/lib to LD_LIBRARY_PATH."
PYTHONPATH="$script_dir/../.." "$python_bin" -m \
  integrations.rlbench.rlbench_dynamac.core.pytracik_dependency --allow-temporary-build
