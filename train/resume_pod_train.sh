#!/usr/bin/env bash
# Recovery: wake the training pod, reinstall deps (the image layer is
# ephemeral, the volume is not), and resume from the latest checkpoint
# under /workspace/train-output/full-ft/.
#
# Pod start uses REST v2 POST /pods/{id}/action {"action":"start"}.
# REST v1 POST /pods/{id}/start is retired.
#
# If this run moves the pod into a compute status and training launch is
# not confirmed, the script sends {"action":"stop"} and waits for EXITED
# or TERMINATED. It never sends terminate. A pod that is already RUNNING
# is left running when a later step fails.
#
# Idempotent — safe to re-run if the resume itself gets preempted.
set -euo pipefail

POD_ID="${MM_POD_ID:-}"
if [[ -z "$POD_ID" ]]; then
    echo "MM_POD_ID is required" >&2
    exit 1
fi
SSH_KEY="${MM_SSH_KEY:-$HOME/.ssh/id_ed25519}"
MM_RUNPOD_API_BASE="${MM_RUNPOD_API_BASE:-https://api.runpod.io/v2}"
MM_RUNPOD_API_BASE="${MM_RUNPOD_API_BASE%/}"
MM_RUNPOD_GRAPHQL_URL="${MM_RUNPOD_GRAPHQL_URL:-https://api.runpod.io/graphql}"
MM_RESUME_MAX_POLLS="${MM_RESUME_MAX_POLLS:-60}"
MM_RESUME_POLL_SECONDS="${MM_RESUME_POLL_SECONDS:-10}"
MM_RESUME_STOP_POLLS="${MM_RESUME_STOP_POLLS:-12}"
MM_RESUME_INJECT="${MM_RESUME_INJECT:-}"

if [[ ! "$POD_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "invalid pod id" >&2
    exit 1
fi
if [[ ! "$MM_RESUME_MAX_POLLS" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid MM_RESUME_MAX_POLLS" >&2
    exit 1
fi
if [[ ! "$MM_RESUME_POLL_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "invalid MM_RESUME_POLL_SECONDS" >&2
    exit 1
fi
if [[ ! "$MM_RESUME_STOP_POLLS" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid MM_RESUME_STOP_POLLS" >&2
    exit 1
fi
case "$MM_RESUME_INJECT" in
    ""|fail-after-ready|confirm-launch) ;;
    *)
        echo "invalid MM_RESUME_INJECT" >&2
        exit 1
        ;;
esac
case "$MM_RUNPOD_API_BASE" in
    https://*|http://127.0.0.1:*|http://localhost:*) ;;
    *)
        echo "refusing API base" >&2
        exit 1
        ;;
esac
case "$MM_RUNPOD_GRAPHQL_URL" in
    https://api.runpod.io/graphql|http://127.0.0.1:*|http://localhost:*) ;;
    *)
        echo "refusing graphql url" >&2
        exit 1
        ;;
esac

# Default HF token location — override with HF_TOKEN_FILE env. Kept under
# $HOME (not /tmp) to avoid a world-readable, predictably-named path that
# other tenants of a shared box could race or read. 0600 enforced below.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.config/mind-mem/hf_write_token}"
if [[ ! -f "${HF_TOKEN_FILE}" ]]; then
    echo "missing HF token at ${HF_TOKEN_FILE}" >&2
    echo "create it with: install -m 0600 /dev/stdin '${HF_TOKEN_FILE}' <<<\"hf_...\"" >&2
    exit 1
fi
# Refuse if the file is group/world readable — token would survive any
# ps/lsof leak that argv-injection would have caused.
# Python reports the mode. The GNU and BSD stat flags do not agree.
perms=$(python3 -c 'import os, sys; print(format(os.stat(sys.argv[1]).st_mode & 0o777, "o"))' "${HF_TOKEN_FILE}")
if [[ "${perms}" != "600" && "${perms}" != "400" ]]; then
    echo "refusing to use ${HF_TOKEN_FILE}: perms ${perms}, want 600" >&2
    exit 1
fi

api_key() {
    awk -F'=' '/api_key|^apikey/ {gsub(/[ "]/, "", $2); print $2; exit}' "${RUNPOD_CONFIG:-$HOME/.runpod/config.toml}"
}

RUNPOD_API_KEY="$(api_key)"
if [[ -z "$RUNPOD_API_KEY" ]]; then
    echo "no api_key in ${RUNPOD_CONFIG:-$HOME/.runpod/config.toml}" >&2
    exit 1
fi

STARTED_BY_US=0
LAUNCH_CONFIRMED=0
STOPPING=0
PS_STATUS=""
PS_HOST=""
PS_PORT=""
PS_ACTIONS=""
PS_LOCKED=""

parse_line() {
    PS_STATUS=""
    PS_HOST=""
    PS_PORT=""
    PS_ACTIONS=""
    PS_LOCKED=""
    IFS=$'\t' read -r PS_STATUS PS_HOST PS_PORT PS_ACTIONS PS_LOCKED <<<"${1-}" || true
    # Bash read collapses consecutive tab delimiters, so empty fields are "-".
    [[ "$PS_HOST" == "-" ]] && PS_HOST=""
    [[ "$PS_PORT" == "-" ]] && PS_PORT=""
    [[ "$PS_ACTIONS" == "-" ]] && PS_ACTIONS=""
    [[ -n "$PS_STATUS" ]]
}

# Print one TSV row: status, ssh host, ssh port, actions, locked.
# Problem responses go to stderr and return non-zero. The pod env is not printed.
runpod_request() {
    local method="$1" path="$2" data="${3-}"
    local tmp code rc
    tmp="$(mktemp)"
    code="000"
    if [[ -n "$data" ]]; then
        code="$(curl -sS -m 60 -o "$tmp" -w '%{http_code}' -X "$method" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -H 'Content-Type: application/json' \
            --data "$data" \
            "${MM_RUNPOD_API_BASE}${path}")" || code="000"
    else
        code="$(curl -sS -m 60 -o "$tmp" -w '%{http_code}' -X "$method" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -H 'Content-Type: application/json' \
            "${MM_RUNPOD_API_BASE}${path}")" || code="000"
    fi
    rc=0
    python3 - "$code" "$tmp" "$POD_ID" <<'PY' || rc=$?
import json
import re
import sys

code_text, path, pod_id = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    code = int(code_text)
except ValueError:
    code = 0
raw = open(path, encoding="utf-8", errors="replace").read()
try:
    body = json.loads(raw) if raw.strip() else {}
except json.JSONDecodeError:
    body = {}
if not isinstance(body, dict):
    body = {}

def fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)

if code < 200 or code >= 300:
    title = body.get("title") if isinstance(body.get("title"), str) else "HTTP error"
    detail = body.get("detail") if isinstance(body.get("detail"), str) else raw[:400]
    errors = body.get("errors") if isinstance(body.get("errors"), list) else []
    extra = ""
    shown = [str(item) for item in errors if isinstance(item, str)]
    if shown:
        extra = "; " + "; ".join(shown)
    http_status = body.get("status") if isinstance(body.get("status"), int) else code
    fail(f"{title} ({http_status}): {detail}{extra}")

if body.get("id") not in (None, pod_id):
    fail("pod id mismatch")
status = body.get("status")
if not isinstance(status, str) or re.fullmatch(r"[A-Z_]+", status) is None:
    fail("pod response has no status")

host = ""
port = ""
direct = (body.get("ssh") or {}).get("direct") if isinstance(body.get("ssh"), dict) else None
if isinstance(direct, dict):
    host = direct.get("host") if isinstance(direct.get("host"), str) else ""
    port_value = direct.get("port")
    if isinstance(port_value, int) and not isinstance(port_value, bool) and 1 <= port_value <= 65535:
        port = str(port_value)
runtime = body.get("runtime") if isinstance(body.get("runtime"), dict) else {}
if (not host or not port) and isinstance(runtime, dict):
    for item in runtime.get("ports") or []:
        if not isinstance(item, dict):
            continue
        private = item.get("private")
        public = item.get("public")
        ip = item.get("ip")
        if (
            private == 22
            and isinstance(ip, str)
            and isinstance(public, int)
            and not isinstance(public, bool)
            and 1 <= public <= 65535
        ):
            host = ip
            port = str(public)
            break
if host and re.fullmatch(r"[0-9A-Za-z.:-]+", host) is None:
    host = ""
if port and re.fullmatch(r"[1-9][0-9]*", port) is None:
    port = ""
actions = body.get("actions") if isinstance(body.get("actions"), list) else []
safe_actions = [item for item in actions if isinstance(item, str) and re.fullmatch(r"[a-z]+", item)]
locked = "1" if body.get("locked") is True else "0"
action_field = ",".join(safe_actions) or "-"
print(f"{status}\t{host or '-'}\t{port or '-'}\t{action_field}\t{locked}")
PY
    rm -f "$tmp"
    return "$rc"
}

graphql_ssh() {
    local tmp rc
    tmp="$(mktemp)"
    curl -sS -m 30 -o "$tmp" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -H 'Content-Type: application/json' \
        -H 'User-Agent: Mozilla/5.0' \
        --data '{"query":"{ myself { pods { id runtime { ports { ip privatePort publicPort isIpPublic } } } } }"}' \
        "$MM_RUNPOD_GRAPHQL_URL" || {
        rm -f "$tmp"
        return 1
    }
    rc=0
    python3 - "$POD_ID" "$tmp" <<'PY' || rc=$?
import json
import re
import sys

pod_id, path = sys.argv[1], sys.argv[2]
try:
    body = json.load(open(path, encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    sys.exit(1)
pods = ((body.get("data") or {}).get("myself") or {}).get("pods") or []
if not isinstance(pods, list):
    sys.exit(1)
for pod in pods:
    if not isinstance(pod, dict) or pod.get("id") != pod_id:
        continue
    runtime = pod.get("runtime") if isinstance(pod.get("runtime"), dict) else {}
    for item in runtime.get("ports") or []:
        if not isinstance(item, dict):
            continue
        if item.get("privatePort") != 22 or item.get("isIpPublic") is not True:
            continue
        host = item.get("ip")
        public = item.get("publicPort")
        if (
            not isinstance(host, str)
            or not isinstance(public, int)
            or isinstance(public, bool)
            or not 1 <= public <= 65535
        ):
            continue
        port = str(public)
        if re.fullmatch(r"[0-9A-Za-z.:-]+", host) and re.fullmatch(r"[1-9][0-9]*", port):
            print(f"{host} {port}")
            sys.exit(0)
sys.exit(1)
PY
    rm -f "$tmp"
    return "$rc"
}

confirm_stopped() {
    local line="" i
    runpod_request POST "/pods/${POD_ID}/action" '{"action":"stop"}' >/dev/null || true
    for i in $(seq 1 "$MM_RESUME_STOP_POLLS"); do
        if line="$(runpod_request GET "/pods/${POD_ID}")"; then
            parse_line "$line" || true
            case "$PS_STATUS" in
                EXITED)
                    echo "pod ${POD_ID} status EXITED. Compute is not billed. The volume can still be billed."
                    STARTED_BY_US=0
                    return 0
                    ;;
                TERMINATED)
                    echo "pod ${POD_ID} status TERMINATED."
                    STARTED_BY_US=0
                    return 0
                    ;;
            esac
        fi
        if [[ "$MM_RESUME_POLL_SECONDS" != 0 ]]; then
            sleep "$MM_RESUME_POLL_SECONDS"
        fi
    done
    echo "pod ${POD_ID} status ${PS_STATUS:-unknown}. Compute may still be billed." >&2
    return 1
}

on_exit() {
    local rc=$?
    trap - EXIT
    set +e
    if [[ "$STARTED_BY_US" == 1 && "$LAUNCH_CONFIRMED" != 1 && "$STOPPING" != 1 ]]; then
        STOPPING=1
        confirm_stopped || rc=1
    fi
    exit "$rc"
}
trap on_exit EXIT

echo "=== Step 1: start pod ${POD_ID} ==="
line="$(runpod_request GET "/pods/${POD_ID}")"
parse_line "$line"
echo "pod status ${PS_STATUS}"
case "$PS_STATUS" in
    RUNNING)
        echo "pod is already RUNNING. A failed resume will not stop it."
        ;;
    STARTING|PROVISIONING)
        echo "pod is ${PS_STATUS}. A failed resume will stop it."
        STARTED_BY_US=1
        ;;
    EXITED)
        # Own the pod as soon as start is attempted. A failed POST can still
        # have moved it into a compute status. A later read of EXITED or
        # TERMINATED clears this flag.
        STARTED_BY_US=1
        if ! line="$(runpod_request POST "/pods/${POD_ID}/action" '{"action":"start"}')"; then
            if line="$(runpod_request GET "/pods/${POD_ID}")"; then
                parse_line "$line" || true
                case "$PS_STATUS" in
                    EXITED|TERMINATED)
                        STARTED_BY_US=0
                        ;;
                esac
            fi
            exit 1
        fi
        parse_line "$line" || true
        echo "start accepted. status ${PS_STATUS:-unknown}"
        ;;
    ERROR)
        echo "pod status ERROR. Not sending start." >&2
        if [[ "$PS_LOCKED" == 1 ]]; then
            echo "pod ${POD_ID} is locked. Compute may still be billed." >&2
            exit 1
        fi
        if [[ ",${PS_ACTIONS}," == *",stop,"* ]]; then
            STARTED_BY_US=1
        else
            echo "pod ${POD_ID} status ERROR has no stop action. Compute may still be billed." >&2
        fi
        exit 1
        ;;
    TERMINATED)
        echo "pod status TERMINATED" >&2
        exit 1
        ;;
    *)
        echo "unexpected pod status ${PS_STATUS}" >&2
        exit 1
        ;;
esac

echo ""
echo "=== Step 2: wait for SSH ==="
ip=""
port=""
for _ in $(seq 1 "$MM_RESUME_MAX_POLLS"); do
    line="$(runpod_request GET "/pods/${POD_ID}")"
    parse_line "$line"
    case "$PS_STATUS" in
        ERROR)
            echo "pod status ERROR" >&2
            exit 1
            ;;
        TERMINATED)
            STARTED_BY_US=0
            echo "pod status TERMINATED" >&2
            exit 1
            ;;
        EXITED)
            STARTED_BY_US=0
            echo "pod status EXITED before SSH was ready" >&2
            exit 1
            ;;
        PROVISIONING|STARTING|RUNNING) ;;
        *)
            echo "unexpected pod status ${PS_STATUS}" >&2
            exit 1
            ;;
    esac
    if [[ -n "$PS_HOST" && -n "$PS_PORT" ]]; then
        ip="$PS_HOST"
        port="$PS_PORT"
        break
    fi
    if [[ "$PS_STATUS" == RUNNING ]]; then
        gql=""
        if gql="$(graphql_ssh)"; then
            read -r ip port <<<"$gql"
            if [[ -n "$ip" && -n "$port" ]]; then
                break
            fi
        fi
        ip=""
        port=""
    fi
    echo -n "."
    if [[ "$MM_RESUME_POLL_SECONDS" != 0 ]]; then
        sleep "$MM_RESUME_POLL_SECONDS"
    fi
done
if [[ -z "$ip" || -z "$port" ]]; then
    echo "pod failed to expose SSH" >&2
    exit 1
fi
echo "ssh ready: ${ip}:${port}"

if [[ "$MM_RESUME_INJECT" == fail-after-ready ]]; then
    echo "injected failure after SSH endpoint was ready" >&2
    exit 1
fi
if [[ "$MM_RESUME_INJECT" == confirm-launch ]]; then
    LAUNCH_CONFIRMED=1
    echo "injected launch confirmation"
    exit 0
fi

SSH_OPTS=(-i "${SSH_KEY}" -p "${port}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20)

echo ""
echo "=== Step 3: reinstall deps ==="
ssh "${SSH_OPTS[@]}" "root@${ip}" "pip install --no-cache-dir 'transformers==5.7.0' 'trl==1.2.0' 'peft==0.14.0' 'accelerate==1.5.0' 'bitsandbytes==0.46.1' 'datasets>=4.7.0,<5' huggingface_hub 2>&1 | tail -3"

echo ""
echo "=== Step 4: detect resume state ==="
ssh "${SSH_OPTS[@]}" "root@${ip}" "test -s /workspace/_causal_lm_import.py && test -s /workspace/mind_mem_causal_lm_loader.py" || {
    echo "missing deployed causal loader bundle; refusing to launch stale training script" >&2
    echo "redeploy with train/runpod_deploy.py so both loader files reach /workspace" >&2
    exit 1
}
ckpt=$(ssh "${SSH_OPTS[@]}" "root@${ip}" "ls -d /workspace/train-output/full-ft/checkpoint-* 2>/dev/null | tail -1")
if [[ -n "$ckpt" ]]; then
    echo "found checkpoint at $ckpt — will resume"
else
    echo "no checkpoint — fresh start"
fi

echo ""
echo "=== Step 5: relaunch training ==="
# Token is piped over ssh stdin and read by the remote shell — it never
# appears in argv (so `ps`/`auditd`/`/proc/*/cmdline` on either side stays
# clean) and it's never assigned to a local shell variable (so no shell
# history risk on this host either).
ssh "${SSH_OPTS[@]}" "root@${ip}" 'set -e
IFS= read -r HF_TOKEN
cd /workspace
test -f train-output/train.log && mv train-output/train.log train-output/train-prev.log
HF_TOKEN="${HF_TOKEN}" MM_TRAIN_ROOT=/workspace/train-output MM_CORPUS=/workspace/train-output/corpus.jsonl \
    nohup python3 -u runpod_full_ft.py >/workspace/train-output/train.log 2>&1 </dev/null &
echo "launched pid=$!"
sleep 3
head -10 /workspace/train-output/train.log' < "${HF_TOKEN_FILE}"
LAUNCH_CONFIRMED=1

echo ""
echo "=== Resume launched at ${ip}:${port} ==="
echo "Tail with:"
echo "  ssh -i ${SSH_KEY} -p ${port} -o StrictHostKeyChecking=no root@${ip} 'tail -F /workspace/train-output/train.log'"
