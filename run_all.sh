#!/bin/bash
# Queue of every remaining job (Milestones 2-7), in order. Each job writes its own log under results/logs/.
# Independent jobs are chained with ';' (one failure does not block the rest); the editor campaigns need the dry run.
cd "$(dirname "$0")"
mkdir -p results/logs
L=results/logs
step() { echo "$(date '+%F %T') START $1" >> $L/queue.log; shift; "$@"; echo "$(date '+%F %T') END rc=$?" >> $L/queue.log; }
R="uv run python"

# M4 dry run was run on its own (uv run python editors.py dry > results/logs/m4_dryrun.log); campaigns need it to pass
[ -f $L/m4_dryrun.log ] || step "M4 dry run" $R editors.py dry > $L/m4_dryrun.log 2>&1
DRY_OK=$(grep -c '"rebuild_identical": true' $L/m4_dryrun.log)
done_set() { [ -f "$1/heldout_summary.json" ] && grep -q "\"$2\"" "$1/heldout_summary.json"; }   # skip finished evals
done_set results/base/heldout ifeval || step "base ifeval" $R evaluate.py run base results/base/heldout --sets ifeval > $L/base_ifeval.log 2>&1
{ done_set results/baselines/scripted/heldout math500 && done_set results/baselines/random/heldout math500; } || \
  step "M2 scripted+random"  $R baselines.py scripted                                 > $L/m2_scripted.log 2>&1
if [ "$DRY_OK" -ge 1 ]; then
  step "M5 campaign 4B"      $R editors.py campaign 4B                                >> $L/m5_campaign_4B.log 2>&1
else
  echo "$(date '+%F %T') SKIP campaigns: dry run did not confirm rebuild" >> $L/queue.log
fi
step "M2 scripted ifeval"    $R baselines.py heldout scripted ifeval                  > $L/scripted_ifeval.log 2>&1
step "M2 random ifeval"      $R baselines.py heldout random ifeval                    > $L/random_ifeval.log 2>&1
for b in mathneuro lora; do
  step "M6 $b heldout"       $R baselines.py heldout $b math500,ifeval                > $L/m6_${b}_heldout.log 2>&1
done
for b in scripted random mathneuro lora; do
  step "KL $b"               $R evaluate.py kl results/baselines/$b/final results/baselines/$b/kl.json > $L/kl_$b.log 2>&1
done
if [ "$DRY_OK" -ge 1 ]; then
  step "M5 campaign 9B"      $R editors.py campaign 9B                                >> $L/m5_campaign_9B.log 2>&1
fi
step "base gsm8k_test"       $R evaluate.py run base results/base/heldout --sets gsm8k_test > $L/base_gsm8k.log 2>&1
for b in scripted random mathneuro lora; do
  step "gsm8k $b"            $R baselines.py heldout $b gsm8k_test                    > $L/gsm8k_$b.log 2>&1
done
step "M7 analyze"            $R analyze.py                                            > $L/m7_analyze.log 2>&1
echo "$(date '+%F %T') QUEUE DONE" >> $L/queue.log
