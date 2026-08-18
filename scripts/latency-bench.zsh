#!/usr/bin/env zsh
# Wall-clock latency for gllm: question sent -> full answer in hand.
#
# Measures what a caller actually feels on a one-shot invocation: process
# start, request, generation, exit. Nothing is warmed by a previous run —
# gllm never requests caching, so every call pays full freight.
#
#   ./latency-bench.zsh                                  # 3 runs, luna vs ds-flash
#   RUNS=7 ./latency-bench.zsh                           # more samples
#   REASONING=low ./latency-bench.zsh                    # with thinking on
#   ./latency-bench.zsh gpt-5.6-luna gemini-3.5-flash    # pick your own
#
# DeepSeek's rate depends on the clock, so the table reports the price_window
# gllm itself resolved — you never have to remember whether you ran it in a
# peak hour. To compare peak against off-peak, run it twice: once inside
# 01:00-04:00 or 06:00-10:00 UTC, once outside.

set -euo pipefail
zmodload zsh/datetime

RUNS=${RUNS:-3}
REASONING=${REASONING:-}
# NOT ${@:-a b} — that expands the default as ONE word and benchmarks a
# model named "gpt-5.6-luna deepseek-v4-flash".
if (( $# )); then MODELS=($@); else MODELS=(gpt-5.6-luna deepseek-v4-flash); fi

command -v gllm >/dev/null || { print -u2 "no gllm on PATH"; exit 1 }
command -v jq   >/dev/null || { print -u2 "no jq on PATH"; exit 1 }

# Two shapes, because they measure different things: `short` is dominated by
# round-trip and startup, `long` by generation throughput.
local -A PROMPTS=(
  short "Reply with exactly one word: pong"
  long  "Explain what a B-tree is and why databases use one instead of a binary search tree. Around 200 words."
)

median() {  # median of the numbers on stdin
  local -a v=(${(f)"$(sort -n)"})
  print -- $v[$(( (${#v} + 1) / 2 ))]
}

print -- "runs=$RUNS  reasoning=${REASONING:-none}  started $(date -u '+%H:%M UTC')"
printf '\n%-20s %-6s %8s %8s %8s %7s %8s  %s\n' \
  MODEL SHAPE MEDIAN_s MIN_s MAX_s OUT_TOK TOK/s WINDOW

for model in $MODELS; do
  for shape in short long; do
    local -a times=()
    local out_tokens="" window="" answer="" rec="" err=""
    err=$(mktemp) || exit 1

    for _ in {1..$RUNS}; do
      local -a flags=(--usage -m $model)
      [[ -n $REASONING ]] && flags+=(-r $REASONING)

      local t0=$EPOCHREALTIME
      # Failure is fatal on purpose: a benchmark that silently averages in a
      # failed call is worse than no benchmark.
      answer=$(gllm $flags -- "$PROMPTS[$shape]" 2>$err) || {
        print -u2 "\ngllm failed for $model/$shape:"; cat $err >&2; exit 1
      }
      times+=( $(( EPOCHREALTIME - t0 )) )

      rec=$(grep '^gllm-usage ' $err | tail -1 | cut -c12-)
      out_tokens=$(jq -r '.output_tokens // 0' <<<"$rec")
      window=$(jq -r '.price_window // "-"' <<<"$rec")
    done
    rm -f $err

    local med=$(print -l $times | median)
    local lo=$(print -l $times | sort -n | head -1)
    local hi=$(print -l $times | sort -n | tail -1)
    local tps=$(( out_tokens / med ))

    printf '%-20s %-6s %8.2f %8.2f %8.2f %7d %8.1f  %s\n' \
      $model $shape $med $lo $hi $out_tokens $tps $window
  done
done
