#!/usr/bin/env bash
# Run gr00t/eval/rollout_policy.py once per LIBERO Object-suite task (10 tasks)
# against an already-running policy server, and print a pooled success rate
# across the whole suite (matching the "197/200" convention used in
# examples/LIBERO/README.md).
#
# Prereqs:
#   - A policy server must already be running (see examples/LIBERO/README.md,
#     "Evaluate checkpoint" section):
#       uv run python gr00t/eval/run_gr00t_server.py \
#           --model-path <checkpoint_dir> \
#           --embodiment-tag LIBERO_PANDA \
#           --use-sim-policy-wrapper
#   - The LIBERO sim venv must be set up (bash gr00t/eval/sim/LIBERO/setup_libero.sh).
#
# Usage:
#   bash gr00t/eval/sim/LIBERO/run_object_suite.sh [--log-dir <path>]
#
# Env vars (all optional, matching rollout_policy.py's own flags):
#   POLICY_CLIENT_HOST (default 127.0.0.1)
#   POLICY_CLIENT_PORT (default 5555)
#   N_EPISODES          (default 20, per task -> 200 total across the suite)
#   N_ENVS              (default 8)
#   N_ACTION_STEPS      (default 8)
#   MAX_EPISODE_STEPS   (default 720)
#   POLICY_CLIENT_TIMEOUT_MS (default 60000)

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../../.."
LIBERO_PYTHON="$SCRIPT_DIR/libero_uv/.venv/bin/python"

POLICY_CLIENT_HOST="${POLICY_CLIENT_HOST:-127.0.0.1}"
POLICY_CLIENT_PORT="${POLICY_CLIENT_PORT:-5555}"
N_EPISODES="${N_EPISODES:-20}"
N_ENVS="${N_ENVS:-8}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
POLICY_CLIENT_TIMEOUT_MS="${POLICY_CLIENT_TIMEOUT_MS:-60000}"
LOG_DIR="eval_logs/libero_object_$(date +%Y%m%d_%H%M%S)"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Official 10 tasks of the LIBERO Object suite (examples/LIBERO/README.md).
TASKS=(
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    # "pick_up_the_cream_cheese_and_place_it_in_the_basket"
    # "pick_up_the_salad_dressing_and_place_it_in_the_basket"
    # "pick_up_the_bbq_sauce_and_place_it_in_the_basket"
    # "pick_up_the_ketchup_and_place_it_in_the_basket"
    # "pick_up_the_tomato_sauce_and_place_it_in_the_basket"
    # "pick_up_the_butter_and_place_it_in_the_basket"
    # "pick_up_the_milk_and_place_it_in_the_basket"
    # "pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
    # "pick_up_the_orange_juice_and_place_it_in_the_basket"
)

mkdir -p "$LOG_DIR"
echo "Logs -> $LOG_DIR"
echo "Server: tcp://$POLICY_CLIENT_HOST:$POLICY_CLIENT_PORT | n_episodes/task=$N_EPISODES n_envs=$N_ENVS"
echo

declare -a TASK_NAMES
declare -a TASK_RATES
FAILED_TASKS=()
TOTAL_SUCCESSES=0
TOTAL_EPISODES=0

for task in "${TASKS[@]}"; do
    log_file="$LOG_DIR/${task}.log"
    video_dir="$LOG_DIR/videos/${task}"
    echo "=== $task ==="

    "$LIBERO_PYTHON" "$PROJECT_ROOT/gr00t/eval/rollout_policy.py" \
        --n-episodes "$N_EPISODES" \
        --policy-client-host "$POLICY_CLIENT_HOST" \
        --policy-client-port "$POLICY_CLIENT_PORT" \
        --video-dir "$video_dir" \
        --max-episode-steps "$MAX_EPISODE_STEPS" \
        --env-name "libero_sim/$task" \
        --n-action-steps "$N_ACTION_STEPS" \
        --n-envs "$N_ENVS" \
        --policy-client-timeout-ms "$POLICY_CLIENT_TIMEOUT_MS" \
        > "$log_file" 2>&1
    status=$?

    rate_line=$(grep -oE "success rate:\s*[0-9.]+" "$log_file" | tail -1)
    rate=$(echo "$rate_line" | grep -oE "[0-9.]+$")

    if [ "$status" -ne 0 ] || [ -z "$rate" ]; then
        echo "  FAILED (exit=$status) -- see $log_file"
        FAILED_TASKS+=("$task")
        continue
    fi

    successes=$(python3 -c "print(round($rate * $N_EPISODES))")
    echo "  success rate: $rate  ($successes/$N_EPISODES)"

    TASK_NAMES+=("$task")
    TASK_RATES+=("$rate")
    TOTAL_SUCCESSES=$((TOTAL_SUCCESSES + successes))
    TOTAL_EPISODES=$((TOTAL_EPISODES + N_EPISODES))
done

echo
echo "==================== SUMMARY ===================="
for i in "${!TASK_NAMES[@]}"; do
    printf "%-70s %s\n" "${TASK_NAMES[$i]}" "${TASK_RATES[$i]}"
done
if [ "${#FAILED_TASKS[@]}" -gt 0 ]; then
    echo
    echo "Failed tasks (excluded from pooled rate):"
    for t in "${FAILED_TASKS[@]}"; do
        echo "  - $t"
    done
fi
echo
if [ "$TOTAL_EPISODES" -gt 0 ]; then
    python3 -c "print(f'Pooled success rate: {$TOTAL_SUCCESSES}/{$TOTAL_EPISODES} ({100*$TOTAL_SUCCESSES/$TOTAL_EPISODES:.2f}%)')"
else
    echo "No tasks completed successfully."
fi
