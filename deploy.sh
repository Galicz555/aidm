#!/usr/bin/env bash
# Deploy the current `production` branch to the AIDM VPS.
#
# Usage:
#   ./deploy.sh                       # uses $AIDM_VPS_HOST (default: aidm-prod)
#   AIDM_VPS_HOST=root@203.0.113.10 ./deploy.sh
#
# Prereqs:
#   - You've already merged your changes from master into production locally
#     and pushed them to GitHub.
#   - ssh access to the VPS works (key + ssh config).
#   - The VPS has been bootstrapped (see deploy/README.md).
#
# What this does on the VPS:
#   1. cd /opt/aidm
#   2. git fetch + checkout production + fast-forward pull
#   3. npm install + npm run build in web/
#   4. docker compose -f docker-compose.prod.yml up -d --build
#   5. docker compose ps  (so you see the post-deploy state)
#
# Volumes (/srv/aidm-vault, ./data, ./certs) survive across deploys.

set -euo pipefail

HOST="${AIDM_VPS_HOST:-aidm-prod}"
REPO_DIR="${AIDM_REPO_DIR:-/opt/aidm}"

echo "Deploying production → $HOST:$REPO_DIR"

ssh -T "$HOST" bash -se <<EOF
set -euo pipefail
cd "$REPO_DIR"

echo "==> git pull"
git fetch origin production
git checkout production
git pull --ff-only

echo "==> web build"
cd web
npm install --no-audit --no-fund
npm run build
cd ..

echo "==> docker compose up"
docker compose -f docker-compose.prod.yml up -d --build

echo "==> status"
docker compose -f docker-compose.prod.yml ps
EOF

echo
echo "Deploy complete. Tail logs with:"
echo "  ssh $HOST 'docker compose -f $REPO_DIR/docker-compose.prod.yml logs -f --tail=50 orchestrator'"
