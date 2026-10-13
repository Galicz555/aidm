.DEFAULT_GOAL := help

UV    ?= uv
RUFF  := $(UV) run ruff
MYPY  := $(UV) run mypy
PYTEST := $(UV) run pytest

.PHONY: help install sync web-install start-public-voice start-public-voice-debug start-private-text start-private-text-debug stop fmt lint typecheck test check clean audio-devices mic-test vault-lint web-check up-private up-private-text down-private logs-private gateway-profile bench-tools bench-regress bench-regress-smoke loop-sim-smoke loop-sim-10 loop-sim-30 loop-sim-50 judge-run models-pull build up down logs ps

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# --- Core Development ---

install: sync ## Alias for sync

sync: ## Resolve & install dev dependencies (root + all workspace members)
	$(UV) sync --all-extras --all-packages

web-install: ## Install web client dependencies (npm install in web/)
	npm --prefix web install

start-public-voice: ## Start orchestrator (public mode, webrtc) + vite dev server. Use 'make start-public-voice DEBUG=1' for verbose logging.
	@if [ ! -d web/node_modules ]; then \
		echo "web/node_modules missing — run 'make web-install' first"; \
		exit 1; \
	fi
	@./scripts/dev-start.sh public webrtc $(DEBUG)

start-public-voice-debug: ## Start orchestrator (public mode, webrtc) + vite dev server with DEBUG logging
	@if [ ! -d web/node_modules ]; then \
		echo "web/node_modules missing — run 'make web-install' first"; \
		exit 1; \
	fi
	@./scripts/dev-start.sh public webrtc 1

start-private-text: ## Start orchestrator (private mode, text) + vite dev server. Uses localhost llm-gateway. Use 'make start-private-text DEBUG=1' for verbose logging.
	@if [ ! -d web/node_modules ]; then \
		echo "web/node_modules missing — run 'make web-install' first"; \
		exit 1; \
	fi
	@./scripts/dev-start.sh private text $(DEBUG)

start-private-text-debug: ## Start orchestrator (private mode, text) + vite dev server with DEBUG logging
	@if [ ! -d web/node_modules ]; then \
		echo "web/node_modules missing — run 'make web-install' first"; \
		exit 1; \
	fi
	@./scripts/dev-start.sh private text 1

stop: ## Stop the development server and all docker compose services
	@echo "==> stopping all services..."
	@# Kill by port — most reliable, catches the real aidm process even when launched via uv
	@fuser -k 8000/tcp 2>/dev/null || true
	@fuser -k 5173/tcp 2>/dev/null || true
	@# Backup: kill by process name (covers cases where the port already freed but the process lingers)
	@pkill -f "\.venv/bin/aidm" 2>/dev/null || true
	@pkill -f "uv run aidm" 2>/dev/null || true
	@pkill -f "npm.*web.*run dev" 2>/dev/null || true
	@pkill -f "node.*vite" 2>/dev/null || true
	@docker compose --profile private down 2>/dev/null || true
	@docker compose --profile public down 2>/dev/null || true
	@echo "==> done"

# --- Code Quality ---

fmt: ## Format code with ruff
	$(RUFF) format .
	$(RUFF) check --fix-only --exit-zero .

lint: ## Lint with ruff (format check + rule check)
	$(RUFF) format --check .
	$(RUFF) check .

typecheck: ## Run mypy
	$(MYPY)

test: ## Run pytest
	$(PYTEST)

check: lint typecheck test vault-lint web-check ## Run everything CI would run (python + web)

clean: ## Remove caches and build artifacts
	rm -rf .ruff_cache .mypy_cache .pytest_cache build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# --- Validation ---

vault-lint: ## Validate the vault against the schema (services/orchestrator/aidm/vault_schema.py)
	$(UV) run python -m services.orchestrator.scripts.lint_vault

web-check: ## Type-check, unit-test and build the web client (svelte-check + vitest + vite build)
	@if [ ! -d web/node_modules ]; then \
		echo "web/node_modules missing — run 'make web-install' first"; \
		exit 1; \
	fi
	npm --prefix web run check
	npm --prefix web run test
	npm --prefix web run build

# --- Debugging ---

audio-devices: ## List local audio devices (pick indices for AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE)
	$(UV) run python -m aidm.devices

mic-test: ## Record 3s from AUDIO_INPUT_DEVICE and report whether PyAudio actually captured anything
	$(UV) run python -m aidm.mic_test

# --- Docker / Private Mode ---

up-private: ## Bring the full private (local-models) stack up. Requires NVIDIA Container Toolkit.
	docker compose --profile private up -d --build

up-private-text: ## Bring up only llm-gateway (text mode doesn't need STT/TTS).
	docker compose --profile private up -d --build --no-deps llm-gateway

down-private: ## Bring the private stack down
	docker compose --profile private down

logs-private: ## Tail private-profile logs
	docker compose --profile private logs -f --tail=100

# --- Specialized ---

# Model profiles (services/llm-gateway/profiles/<name>.env): PROFILE=<name>
# exports the file's vars into the recipe shell, where they beat .env for both
# compose substitution (gateway) and aidm/config.py (orchestrator).
PROFILES_DIR = services/llm-gateway/profiles
PROFILE_ENV = $(if $(PROFILE),test -f $(PROFILES_DIR)/$(PROFILE).env && set -a && . ./$(PROFILES_DIR)/$(PROFILE).env && set +a &&,)

gateway-profile: ## Recreate llm-gateway with a model profile's gateway vars (PROFILE=qwen3-14b|qwen38-asis|...)
	@test -n "$(PROFILE)" || (echo "usage: make gateway-profile PROFILE=<name>  (ls $(PROFILES_DIR))" && exit 2)
	$(PROFILE_ENV) docker compose --profile private up -d --no-deps --force-recreate llm-gateway

bench-tools: ## Phase 7 gate: vault-tool reliability vs local LLM (expects llm-gateway on localhost:8002). PROFILE=<name> applies a model profile
	$(PROFILE_ENV) $(UV) run python scripts/bench_tool_calls.py $(BENCH_ARGS)

bench-regress: ## Per-fix DM-quality regress (10 replays/scenario, ~30-60 min local LLM)
	$(PROFILE_ENV) $(UV) run python services/orchestrator/scripts/bench_regress.py

bench-regress-smoke: ## Smoke variant: 3 replays/scenario (~10-15 min)
	$(PROFILE_ENV) $(UV) run python services/orchestrator/scripts/bench_regress.py --replays 3

LOOP_SIM = $(PROFILE_ENV) $(UV) run python services/orchestrator/scripts/loop_sim.py
LOOP_SCENARIOS = services/orchestrator/scripts/loop_scenarios

loop-sim-smoke: ## 5-turn closed-loop harness (needs llm-gateway + ANTHROPIC_API_KEY, ~10-15 min)
	$(LOOP_SIM) $(LOOP_SCENARIOS)/smoke-5t.yaml $(LOOP_ARGS)

loop-sim-10: ## 10-turn closed-loop harness (~15-35 min)
	$(LOOP_SIM) $(LOOP_SCENARIOS)/mid-10t.yaml $(LOOP_ARGS)

loop-sim-30: ## 30-turn closed-loop harness (overnight; ~45-90+ min)
	$(LOOP_SIM) $(LOOP_SCENARIOS)/long-30t.yaml $(LOOP_ARGS)

loop-sim-50: ## 50-turn marathon: long-form degradation curve (~25-45 min)
	$(LOOP_SIM) $(LOOP_SCENARIOS)/marathon-50t.yaml $(LOOP_ARGS)

judge-run: ## Score one replay dir: deterministic + Opus judge (REPLAY=docs/bench/.../replay-001)
	@test -n "$(REPLAY)" || (echo "usage: make judge-run REPLAY=docs/bench/<date>/loop/<scenario>/replay-001" && exit 2)
	$(UV) run python services/orchestrator/scripts/judge_run.py $(REPLAY) $(JUDGE_ARGS)

models-pull: ## Fetch GGUF / Whisper / XTTS weights into ./models/ (host-side, before `make up-private`)
	./scripts/pull_models.sh

# --- Public Docker (rarely used directly) ---

build: ## Build the docker images (public profile)
	docker compose --profile public build

up: ## Bring the public stack up (detached, build)
	docker compose --profile public up -d --build

down: ## Bring the public stack down
	docker compose --profile public down

logs: ## Tail public docker compose logs
	docker compose --profile public logs -f --tail=100

ps: ## Show docker compose status (all profiles)
	docker compose ps
