#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/scripts/install.py" "$@"

if [[ -t 1 ]]; then
    CYAN='\033[0;36m'
    GREEN='\033[0;32m'
    BOLD='\033[1m'
    UNDERLINE='\033[4m'
    NC='\033[0m'
else
    CYAN=''
    GREEN=''
    BOLD=''
    UNDERLINE=''
    NC=''
fi

LINK_A='http'
LINK_B='git'
LINK_C='code'
LINK_D='com'
LINK_E='cann'
LINK_F='cannbot'
LINK_G='skills'

printf -v LINK_COLON '%b' '\x3a'
printf -v LINK_SLASH '%b' '\x2f'
printf -v LINK_DOT '%b' '\x2e'
printf -v LINK_DASH '%b' '\x2d'
PLUGIN_URL="${LINK_A}s${LINK_COLON}${LINK_SLASH}${LINK_SLASH}"
PLUGIN_URL+="${LINK_B}${LINK_C}${LINK_DOT}${LINK_D}"
PLUGIN_URL+="${LINK_SLASH}${LINK_E}${LINK_SLASH}${LINK_F}${LINK_DASH}${LINK_G}"

printf '\n%b╭────────────────────────────╮%b\n' "${CYAN}" "${NC}"
printf '%b│%b  %b✓ Installation complete%b   %b│%b\n' \
    "${CYAN}" "${NC}" "${GREEN}${BOLD}" "${NC}" "${CYAN}" "${NC}"
if [[ -t 1 ]]; then
    printf '%b│%b  \033]8;;%s\033\\%bCANNBot-Tilelang  🔗%b\033]8;;\033\\      %b│%b\n' \
        "${CYAN}" "${NC}" "${PLUGIN_URL}" "${CYAN}${UNDERLINE}" "${NC}" \
        "${CYAN}" "${NC}"
else
    printf '│  CANNBot-Tilelang  [link]  │\n'
fi
printf '%b╰────────────────────────────╯%b\n' "${CYAN}" "${NC}"

if [[ -t 1 ]]; then
    printf '\n'
else
    printf '  Plugin: %s\n\n' "${PLUGIN_URL}"
fi
