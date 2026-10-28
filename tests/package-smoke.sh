#!/usr/bin/env bash
# Pack the Tensile package, check what is inside it, then install it into a
# throwaway consumer and run that. CI runs this on every push; the publish
# workflow runs it on the exact package it is about to push.
#
#   tests/package-smoke.sh [version]      # default 0.0.0-local
#
# It exists because the package was broken for as long as nothing installed
# it: pack left Tensile.Kernels.dll out and declared a dependency on a
# "Tensile.Kernels" package that does not exist, and every test passed,
# because every test referenced the projects. Finding 10, one level out:
# a package nothing consumes is not verified by anything.
set -euo pipefail

version="${1:-0.0.0-local}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

out="$root/artifacts/packages"
rm -rf "$out"
mkdir -p "$out"

echo "== pack Tensile $version"
dotnet pack src/Tensile/Tensile.csproj -c Release -o "$out" -p:Version="$version" --nologo

package="$out/Tensile.$version.nupkg"
[[ -f "$package" ]] || { echo "expected $package" >&2; exit 1; }

echo "== contents"
listing="$(unzip -Z1 "$package")"
echo "$listing"
nuspec="$(unzip -p "$package" Tensile.nuspec)"

fail() { echo "FAIL: $*" >&2; exit 1; }

grep -qx "lib/net10.0/Tensile.dll" <<<"$listing" || fail "Tensile.dll is not in the package"
grep -qx "lib/net10.0/Tensile.Kernels.dll" <<<"$listing" || fail "Tensile.Kernels.dll is not in the package"
grep -qx "lib/net10.0/Tensile.xml" <<<"$listing" || fail "the XML documentation is not in the package"
grep -qx "README.md" <<<"$listing" || fail "the README is not in the package"
grep -q "<dependency " <<<"$nuspec" && fail "the package declares a dependency; it should have none"
grep -q "<repository .*commit=" <<<"$nuspec" || fail "the package does not record its source commit"

echo "== install and run in a throwaway consumer"
# A fresh package cache, so the consumer restores this package and not a
# copy of an earlier one with the same version.
cache="$(mktemp -d)"
trap 'rm -rf "$cache"' EXIT

dotnet run --project tests/Tensile.PackageSmoke -c Release --nologo \
    -p:TensilePackageVersion="$version" -p:RestorePackagesPath="$cache"
