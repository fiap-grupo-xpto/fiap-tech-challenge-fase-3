#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'Usage: scripts/scan_secrets_for_codex.sh <file>' >&2; exit 2; }
exec "${SONAR_CLI:-$HOME/.local/share/sonarqube-cli/bin/sonar}" analyze secrets "$1"
