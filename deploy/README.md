# AIDM production deploy

Targets: Hetzner CX23 VPS (Ubuntu 24.04), Cloudflare proxied DNS + Access gate,
public mode only. See the project root plan for the full architecture.

## Layout on the VPS

```
/opt/aidm/                  git clone, production branch checkout
  ├─ docker-compose.prod.yml
  ├─ deploy/
  │   ├─ nginx-bootstrap.conf    port 80 only, used for cert issuance
  │   ├─ nginx.conf              full TLS config, used in normal operation
  │   ├─ nginx-active.conf       symlink/copy of one of the above (gitignored)
  │   ├─ issue-cert.sh
  │   └─ README.md (this file)
  ├─ data/                       runtime state — budget.db + loguru log files
  ├─ certs/                      Let's Encrypt cert tree + ACME challenge root
  ├─ vault/                      seed vault (NOT mounted, see below)
  ├─ web/build/                  SvelteKit static build
  └─ .env.prod                   scp'd manually, chmod 600

/srv/aidm-vault/                 runtime vault (writable, app modifies it)
                                 first deploy: cp -r /opt/aidm/vault /srv/aidm-vault
```

## First-time deploy

Assumes you already have:
- Hetzner VPS up, ssh root login working with `~/.ssh/aidm_ed25519`
- Cloudflare DNS A record for `aidm.<domain>` → VPS IP, proxy ON
- Cloudflare Access app configured, AUD tag + team domain in hand
- GitHub repo cloned/pushed, deploy key on GitHub for the VPS

### 1. Base setup on the VPS

```bash
ssh root@<vps>
apt update && apt install -y ufw
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
# aiortc media plane — wide UDP range Linux uses for ephemeral ports.
ufw allow 49152:65535/udp
ufw --force enable

curl -fsSL https://get.docker.com | sh
```

### 2. Pull the repo + seed vault

```bash
# Generate a deploy key on the VPS and add the .pub to GitHub Settings → Deploy keys
ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N ""
cat ~/.ssh/github_deploy.pub
# (paste into GitHub, then:)
cat >> ~/.ssh/config <<'EOF'
Host github.com
  IdentityFile ~/.ssh/github_deploy
  IdentitiesOnly yes
EOF

mkdir -p /opt && cd /opt
git clone https://github.com/<you>/aidm.git aidm
cd aidm
git checkout production   # create this branch first on your laptop, see below

# Seed runtime vault — one-shot copy. The app writes here from now on.
cp -r vault /srv/aidm-vault

# Frontend build
apt install -y nodejs npm
cd web && npm install && npm run build && cd ..
```

### 3. Secrets

From your laptop:
```bash
scp .env.prod root@<vps>:/opt/aidm/.env.prod
ssh root@<vps> chmod 600 /opt/aidm/.env.prod
```

`.env.prod` must contain at minimum:
```
AIDM_MODE=public
AIDM_TRANSPORT=webrtc
AIDM_ALLOWED_ORIGINS=https://aidm.example.com
AIDM_PUBLIC_IP=<vps-ipv4>
AIDM_MAX_CONCURRENT_SESSIONS=3
AIDM_SESSION_TOKEN_CAP=20000
AIDM_SESSION_TTS_CHAR_CAP=5000
AIDM_DAILY_USD_CAP=0.30
AIDM_RATE_LIMIT_OFFER=5/minute
AIDM_RATE_LIMIT_SESSION=10/minute
CF_ACCESS_TEAM_DOMAIN=your-team.cloudflareaccess.com
CF_ACCESS_AUD=<application-audience-tag>
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
ELEVENLABS_API_KEY=...
```

### 4. Bootstrap nginx (no TLS yet)

```bash
cd /opt/aidm
cp deploy/nginx-bootstrap.conf deploy/nginx-active.conf
docker compose -f docker-compose.prod.yml up -d nginx
```

Verify from your laptop:
```bash
curl http://aidm.<domain>/
# expect: "AIDM bootstrap — cert issuance in progress."
```

### 5. Issue the Let's Encrypt cert

On the VPS:
```bash
cd /opt/aidm
DOMAIN=aidm.example.com EMAIL=you@example.com bash deploy/issue-cert.sh
```

If this fails: Cloudflare's "Always Use HTTPS" or "SSL/TLS → Full (strict)" can
intercept the HTTP-01 challenge. Set SSL/TLS mode to **Full** (not Strict) during
issuance, and disable "Always Use HTTPS" temporarily. Re-enable after success.

### 6. Switch to TLS config

```bash
cp deploy/nginx.conf deploy/nginx-active.conf
docker compose -f docker-compose.prod.yml up -d --build
```

From your laptop, visit `https://aidm.<domain>` — Cloudflare Access gate should
prompt for the OTP, then the SPA loads.

## Day-to-day deploys

From your laptop:
```bash
git checkout master           # work, commit
git checkout production
git merge master
git push origin production
./deploy.sh                   # ssh into the VPS and pull + rebuild
```

`deploy.sh` (repo root) handles `git pull`, `npm run build`, and
`docker compose up -d --build`. The volumes (`/srv/aidm-vault`, `./data`,
`./certs`) survive.

## Cert renewal

The `certbot` service runs a 12h renewal check loop in the background. nginx
does not auto-reload on a fresh cert — but the next `deploy.sh` run will
`up -d --build` which restarts nginx. Cert renews at day 60, expires at day 90,
so a deploy within that 30-day window picks it up. If you go longer:

```bash
ssh root@<vps>
docker compose -f /opt/aidm/docker-compose.prod.yml restart nginx
```

## Pulling session logs back to your laptop

```bash
rsync -av root@<vps>:/srv/aidm-vault/ ~/aidm-vault-prod-mirror/
```

Open the mirror folder as a separate Obsidian vault to read what the DM wrote.

## Rolling back

```bash
ssh root@<vps>
cd /opt/aidm
git log --oneline -10                 # find the SHA you want
git checkout <sha>
docker compose -f docker-compose.prod.yml up -d --build
```
