#!/usr/bin/env bash
# Download the NCI GDC Data Transfer Tool (gdc-client) for Linux x64.
#
# Official reference: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
# Release notes: https://docs.gdc.cancer.gov/Data_Transfer_Tool/Release_Notes/DTT_Release_Notes/
#
# NCI publishes MD5 checksums on the page above. This script verifies the zip
# before extracting. If the default URL fails (Drupal / redirects), set
# GDC_CLIENT_URL to the link you copy from the official page, or install via
#   conda install -c bioconda gdc-client
#
set -euo pipefail

VERSION="${GDC_CLIENT_VERSION:-2.3.0}"
# MD5 for gdc-client_${VERSION}_Ubuntu_x64-py3.8-ubuntu-20.04 per NCI GDC page (v2.3.0)
EXPECTED_MD5="${GDC_CLIENT_MD5:-18591d74de07cdcd396dab71c52663da}"

DEFAULT_URL="https://gdc.cancer.gov/files/public/file/gdc-client_${VERSION}_Ubuntu_x64-py3.8-ubuntu-20.04.zip"
URL="${GDC_CLIENT_URL:-$DEFAULT_URL}"

DEST_DIR="${1:-./tools/gdc-client}"
mkdir -p "$DEST_DIR"
ZIP_PATH="${DEST_DIR}/gdc-client_${VERSION}_linux.zip"

echo "Downloading GDC client from:"
echo "  $URL"
echo "Expected MD5 (Ubuntu x64, v${VERSION}): $EXPECTED_MD5"
echo ""

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi

curl -fL --retry 3 -o "$ZIP_PATH" "$URL"

# Reject HTML error pages and validate ZIP
if ! unzip -t "$ZIP_PATH" >/dev/null 2>&1; then
  echo "error: downloaded file is not a valid ZIP (often an HTML error page if the URL moved)." >&2
  echo "Set GDC_CLIENT_URL to the exact download URL from" >&2
  echo "  https://gdc.cancer.gov/access-data/gdc-data-transfer-tool" >&2
  echo "or use: conda install -c bioconda gdc-client" >&2
  rm -f "$ZIP_PATH"
  exit 1
fi

# Verify (GNU md5sum first, then fall back to md5 on macOS)
if command -v md5sum >/dev/null 2>&1; then
  echo "$EXPECTED_MD5  $ZIP_PATH" | md5sum -c -
elif command -v md5 >/dev/null 2>&1; then
  actual=$(md5 -q "$ZIP_PATH" 2>/dev/null || true)
  if [ "$actual" != "$EXPECTED_MD5" ]; then
    echo "error: MD5 mismatch (expected $EXPECTED_MD5, got $actual)" >&2
    exit 1
  fi
  echo "MD5 OK"
else
  echo "warning: no md5sum/md5; skipping checksum" >&2
fi

unzip -o -q "$ZIP_PATH" -d "$DEST_DIR"
echo ""
echo "Extracted under: $DEST_DIR"
echo "Add the directory containing the 'gdc-client' binary to your PATH."
