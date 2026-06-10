#!/usr/bin/env bash
# Recovery: pod uz2uajluzskmm2 was preempted mid-run. Wake it up,
# reinstall deps (image layer is ephemeral, volume is not), resume
# training from the latest checkpoint under /workspace/train-output/full-ft/.
#
# Idempotent — safe to re-run if the resume itself gets preempted.
set -euo pipefail

POD_ID="uz2uajluzskmm2"
SSH_KEY="${MM_SSH_KEY:-$HOME/.ssh/id_ed25519}"
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
perms=$(stat -c '%a' "${HF_TOKEN_FILE}")
if [[ "${perms}" != "600" && "${perms}" != "400" ]]; then
    echo "refusing to use ${HF_TOKEN_FILE}: perms ${perms}, want 600" >&2
    exit 1
fi

api_key() {
    awk -F'=' '/api_key|^apikey/ {gsub(/[ "]/, "", $2); print $2; exit}' "${RUNPOD_CONFIG:-$HOME/.runpod/config.toml}"
}

echo "=== Step 1: start pod ${POD_ID} ==="
curl -sS -X POST -H "Authorization: Bearer $(api_key)" -H 'Content-Type: application/json' \
     "https://rest.runpod.io/v1/pods/${POD_ID}/start" -d '{}' >/dev/null

echo ""
echo "=== Step 2: wait for SSH ==="
ip=""; port=""
for _ in $(seq 1 60); do
    json=$(curl -sS -H "Authorization: Bearer $(api_key)" -H 'Content-Type: application/json' \
                -H 'User-Agent: Mozilla/5.0' \
                -d "{\"query\":\"{ myself { pods { id desiredStatus runtime { ports { ip privatePort publicPort isIpPublic } } } } }\"}" \
                https://api.runpod.io/graphql)
    ip=$(echo "$json" | python3 -c "import json,sys
b=json.load(sys.stdin)
for p in b.get('data',{}).get('myself',{}).get('pods',[]) or []:
    if p['id']!='${POD_ID}': continue
    rt=p.get('runtime') or {}
    for x in rt.get('ports',[]) or []:
        if x.get('privatePort')==22 and x.get('isIpPublic'):
            print(x['ip']); break
    break")
    port=$(echo "$json" | python3 -c "import json,sys
b=json.load(sys.stdin)
for p in b.get('data',{}).get('myself',{}).get('pods',[]) or []:
    if p['id']!='${POD_ID}': continue
    rt=p.get('runtime') or {}
    for x in rt.get('ports',[]) or []:
        if x.get('privatePort')==22 and x.get('isIpPublic'):
            print(x['publicPort']); break
    break")
    if [[ -n "$ip" && -n "$port" ]]; then
        echo "ssh ready: ${ip}:${port}"
        break
    fi
    echo -n "."
    sleep 10
done
[[ -z "$ip" || -z "$port" ]] && { echo "pod failed to expose SSH"; exit 1; }

SSH_OPTS=(-i "${SSH_KEY}" -p "${port}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20)

echo ""
echo "=== Step 3: reinstall deps ==="
ssh "${SSH_OPTS[@]}" "root@${ip}" "pip install --no-cache-dir 'transformers==5.7.0' 'trl==1.2.0' 'peft==0.14.0' 'accelerate==1.5.0' 'bitsandbytes==0.46.1' 'datasets>=4.7.0,<5' huggingface_hub 2>&1 | tail -3"

echo ""
echo "=== Step 4: detect resume state ==="
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

echo ""
echo "=== Resume launched at ${ip}:${port} ==="
echo "Tail with:"
echo "  ssh -i ${SSH_KEY} -p ${port} -o StrictHostKeyChecking=no root@${ip} 'tail -F /workspace/train-output/train.log'"
