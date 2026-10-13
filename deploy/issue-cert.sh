#!/usr/bin/env bash
# Issue the first Let's Encrypt certificate for the AIDM hostname.
#
# Prereqs:
#   - nginx is running with deploy/nginx-bootstrap.conf (port 80 reachable)
#   - DNS A record points at this VPS, Cloudflare proxy may be on or off
#     (HTTP-01 challenge passes through CF either way)
#
# Usage:
#   DOMAIN=aidm.example.com EMAIL=you@example.com \
#     bash deploy/issue-cert.sh
#
# After this succeeds:
#   cp deploy/nginx.conf  ...                 # the TLS-enabled config
#   docker compose -f docker-compose.prod.yml restart nginx
set -euo pipefail

: "${DOMAIN:?set DOMAIN=aidm.example.com}"
: "${EMAIL:?set EMAIL=you@example.com}"

cd "$(dirname "$0")/.."

docker compose -f docker-compose.prod.yml run --rm certbot \
  certonly --webroot \
  -w /var/www/certbot \
  -d "$DOMAIN" \
  --email "$EMAIL" \
  --agree-tos --no-eff-email \
  --non-interactive

echo
echo "Cert issued. Next:"
echo "  1) Edit deploy/nginx.conf if your domain differs from $DOMAIN."
echo "  2) Bring up the full stack:"
echo "     docker compose -f docker-compose.prod.yml up -d --build"
