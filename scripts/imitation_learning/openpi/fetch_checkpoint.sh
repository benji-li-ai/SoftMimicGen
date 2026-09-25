#!/usr/bin/env bash
# Download one training checkpoint from OCI for local serving/evaluation.
#
# Fetches ONLY params/ and assets/. A full step dir is ~45 GB, of which ~32 GB is train_state/
# (AdamW moments plus EMA) -- needed to resume training, useless for inference. params/ is
# ~12.4 GB and assets/ carries norm_stats.json, which create_trained_policy reads.
#
# Usage:
#   scripts/imitation_learning/openpi/fetch_checkpoint.sh 5000
#   scripts/imitation_learning/openpi/fetch_checkpoint.sh 19999 /tmp/ckpt
set -euo pipefail

STEP="${1:?usage: fetch_checkpoint.sh <step> [dest_root] }"
DEST_ROOT="${2:-/tmp/smg_ckpt}"
RUN="${SMG_RUN:-smg-yam-towel-train20k}"
PREFIX="${SMG_S3_PREFIX:-s3://research-datasets-chicago/users/benji/softmimicgen}/checkpoints/${RUN}"
ENDPOINT="${AWS_ENDPOINT_URL:-https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com}"
REGION="${AWS_REGION:-us-chicago-1}"
PROFILE="${OCI_PROFILE:-oci.chi}"

AWS_ARGS=(--endpoint-url "$ENDPOINT" --region "$REGION" --profile "$PROFILE")
DEST="${DEST_ROOT}/${STEP}"

if ! aws s3 ls "${PREFIX}/${STEP}/" "${AWS_ARGS[@]}" >/dev/null 2>&1; then
    echo "step ${STEP} not present under ${PREFIX} yet. Available:" >&2
    aws s3 ls "${PREFIX}/" "${AWS_ARGS[@]}" 2>/dev/null | awk '$1=="PRE"{print "  " $2}' >&2
    exit 1
fi

mkdir -p "$DEST"
for sub in params assets; do
    echo "downloading ${sub}/ ..."
    aws s3 sync "${PREFIX}/${STEP}/${sub}" "${DEST}/${sub}" "${AWS_ARGS[@]}"
done

# Verify against the remote rather than trusting sync's exit code: OCI's S3 shim has surprised
# us before by skipping objects without erroring.
for sub in params assets; do
    remote=$(aws s3 ls "${PREFIX}/${STEP}/${sub}/" --recursive "${AWS_ARGS[@]}" 2>/dev/null | grep -c .)
    local_n=$(find "${DEST}/${sub}" -type f | wc -l)
    printf '  %-8s remote %3s files, local %3s files' "$sub" "$remote" "$local_n"
    [ "$remote" = "$local_n" ] && echo "  OK" || { echo "  MISMATCH"; exit 1; }
done

echo
echo "checkpoint ready: ${DEST}  ($(du -sh "$DEST" | cut -f1))"
echo "serve it with:"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE=false \\"
echo "    ~/src/openpi/.venv/bin/python scripts/imitation_learning/openpi/serve_checkpoint.py \\"
echo "      --checkpoint_dir ${DEST} --port 8000"
