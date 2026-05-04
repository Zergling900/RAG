#!/usr/bin/env bash

# Start the local services needed by this RAG project.
#
# Run:
#   ./start_rag_stack.sh
#
# Or source it to also activate the Python virtual environment in this shell:
#   source ./start_rag_stack.sh

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  RAG_SCRIPT_SOURCED=1
else
  RAG_SCRIPT_SOURCED=0
  set -euo pipefail
fi

log() {
  printf '[rag-start] %s\n' "$*"
}

warn() {
  printf '[rag-start] WARNING: %s\n' "$*" >&2
}

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
COMPOSE_DIR="$SCRIPT_DIR/docker/milvus"
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"

find_docker_cli() {
  if command -v docker >/dev/null 2>&1; then
    command -v docker
    return 0
  fi

  local app_path
  for app_path in "$HOME/Applications/Docker.app" "/Applications/Docker.app"; do
    if [[ -x "$app_path/Contents/Resources/bin/docker" ]]; then
      printf '%s\n' "$app_path/Contents/Resources/bin/docker"
      return 0
    fi
  done

  return 1
}

find_docker_app() {
  local app_path
  for app_path in "$HOME/Applications/Docker.app" "/Applications/Docker.app"; do
    if [[ -d "$app_path" ]]; then
      printf '%s\n' "$app_path"
      return 0
    fi
  done

  return 1
}

wait_for_docker() {
  local timeout="${DOCKER_WAIT_SECONDS:-180}"
  local elapsed=0

  while (( elapsed < timeout )); do
    if "$DOCKER_BIN" info >/dev/null 2>&1; then
      log "Docker daemon is ready."
      return 0
    fi

    sleep 2
    elapsed=$((elapsed + 2))

    if (( elapsed % 20 == 0 )); then
      log "Waiting for Docker Desktop... ${elapsed}s"
    fi
  done

  warn "Docker Desktop did not become ready within ${timeout}s."
  warn "If a Docker Desktop permission or license prompt is visible, approve it and run this script again."
  return 1
}

start_docker_desktop() {
  if "$DOCKER_BIN" info >/dev/null 2>&1; then
    log "Docker daemon is already running."
    return 0
  fi

  local docker_app
  if ! docker_app="$(find_docker_app)"; then
    warn "Docker Desktop was not found. Install it first, then run this script again."
    return 1
  fi

  log "Opening Docker Desktop: $docker_app"
  open "$docker_app" >/dev/null 2>&1 || true
  wait_for_docker
}

start_milvus() {
  if [[ ! -f "$COMPOSE_DIR/docker-compose.yml" ]]; then
    warn "Milvus compose file was not found: $COMPOSE_DIR/docker-compose.yml"
    return 1
  fi

  log "Starting Milvus Standalone with Docker Compose..."
  (
    cd "$COMPOSE_DIR" || exit 1
    "$DOCKER_BIN" compose up -d
  )
}

wait_for_milvus_health() {
  local timeout="${MILVUS_WAIT_SECONDS:-180}"
  local elapsed=0
  local containers="milvus-etcd milvus-minio milvus-standalone"

  while (( elapsed < timeout )); do
    local all_healthy=1
    local container status

    for container in $containers; do
      status="$("$DOCKER_BIN" inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container" 2>/dev/null || true)"
      if [[ "$status" != "healthy" && "$status" != "running" ]]; then
        all_healthy=0
      fi
    done

    if (( all_healthy == 1 )); then
      log "Milvus services are ready."
      (
        cd "$COMPOSE_DIR" || exit 1
        "$DOCKER_BIN" compose ps
      )
      return 0
    fi

    sleep 2
    elapsed=$((elapsed + 2))

    if (( elapsed % 20 == 0 )); then
      log "Waiting for Milvus health checks... ${elapsed}s"
    fi
  done

  warn "Milvus did not become healthy within ${timeout}s."
  (
    cd "$COMPOSE_DIR" || exit 1
    "$DOCKER_BIN" compose ps
  )
  return 1
}

activate_venv_if_sourced() {
  if (( RAG_SCRIPT_SOURCED == 1 )); then
    if [[ -f "$VENV_ACTIVATE" ]]; then
      # shellcheck source=/dev/null
      source "$VENV_ACTIVATE"
      log "Python virtual environment activated: $SCRIPT_DIR/.venv"
    else
      warn "Virtual environment was not found: $VENV_ACTIVATE"
    fi
  else
    log "Tip: run 'source ./start_rag_stack.sh' if you also want this shell to enter .venv."
  fi
}

main() {
  if ! DOCKER_BIN="$(find_docker_cli)"; then
    warn "docker CLI was not found."
    return 1
  fi
  export DOCKER_BIN

  log "Using docker CLI: $DOCKER_BIN"
  start_docker_desktop || return 1
  start_milvus || return 1
  wait_for_milvus_health || return 1
  activate_venv_if_sourced
}

main "$@"
RAG_SCRIPT_STATUS=$?

if (( RAG_SCRIPT_SOURCED == 1 )); then
  return "$RAG_SCRIPT_STATUS"
fi

exit "$RAG_SCRIPT_STATUS"
