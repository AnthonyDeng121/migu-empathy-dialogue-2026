#!/bin/bash
# Stage list for the unattended generation workload (run_workload.sh).
#
# run_workload.sh sources this file (or $STAGES_FILE) and calls
# workload_stages(), which must print one "name|command" line per stage.
# The name carries a numeric prefix so the marker files in
# $ROOT/stages/<name>.done sort in execution order and a relaunch can tell
# at a glance how far the run got. Commands may not contain a literal "|".
#
# Every path is relative to $ROOT and every interpreter / device / model
# location comes from workload.env, so this file stays host independent.
# Generation stages pin --device cuda and --max-new-tokens 256 to keep the
# arms comparable with the recorded cycle-2 runs; the metric stages are CPU
# only and are ordered after the generations they consume.
workload_stages() {
    cat <<STAGES
01_t4_off_selection|$PY scripts/run_response_inference.py --input $SEL_INPUT --model $MODEL_DIR --adapter $T4_ADAPTER --output $OUT_DIR/t4_longweight_selection_newcard.jsonl --performance-report $OUT_DIR/t4_longweight_selection_newcard_report.json --resume-dir $OUT_DIR/t4_longweight_selection_newcard_progress --max-new-tokens 256 --device cuda
02_d4v2_off_selection|$PY scripts/run_response_inference.py --input $SEL_INPUT --model $MODEL_DIR --adapter $D4V2_ADAPTER --output $OUT_DIR/d4v2_selection_newcard.jsonl --performance-report $OUT_DIR/d4v2_selection_newcard_report.json --resume-dir $OUT_DIR/d4v2_selection_newcard_progress --max-new-tokens 256 --device cuda
03_t4_val|$PY scripts/run_response_inference.py --input $VAL_INPUT --model $MODEL_DIR --adapter $T4_ADAPTER --output $OUT_DIR/t4_longweight_val_newcard.jsonl --performance-report $OUT_DIR/t4_longweight_val_newcard_report.json --resume-dir $OUT_DIR/t4_longweight_val_newcard_progress --max-new-tokens 256 --device cuda
04_d4v2_val|$PY scripts/run_response_inference.py --input $VAL_INPUT --model $MODEL_DIR --adapter $D4V2_ADAPTER --output $OUT_DIR/d4v2_val_newcard.jsonl --performance-report $OUT_DIR/d4v2_val_newcard_report.json --resume-dir $OUT_DIR/d4v2_val_newcard_progress --max-new-tokens 256 --device cuda
05_t4_emotion_selection|$PY scripts/run_response_inference.py --input $SEL_INPUT --model $MODEL_DIR --adapter $T4_ADAPTER --condition-jsonl $C5_CONDITIONS --condition-fields emotion --output $OUT_DIR/t4_emotion_only_selection_newcard.jsonl --performance-report $OUT_DIR/t4_emotion_only_selection_newcard_report.json --resume-dir $OUT_DIR/t4_emotion_only_selection_newcard_progress --max-new-tokens 256 --device cuda
06_t4_eprofile_selection|$PY scripts/run_response_inference.py --input $SEL_INPUT --model $MODEL_DIR --adapter $T4_ADAPTER --condition-jsonl $BOTH_COND_SEL --condition-fields both --output $OUT_DIR/t4_emotion_profile_selection_newcard.jsonl --performance-report $OUT_DIR/t4_emotion_profile_selection_newcard_report.json --resume-dir $OUT_DIR/t4_emotion_profile_selection_newcard_progress --max-new-tokens 256 --device cuda
07_d4v2_emotion_selection|$PY scripts/run_response_inference.py --input $SEL_INPUT --model $MODEL_DIR --adapter $D4V2_ADAPTER --condition-jsonl $C5_CONDITIONS --condition-fields emotion --output $OUT_DIR/d4v2_emotion_only_selection_newcard.jsonl --performance-report $OUT_DIR/d4v2_emotion_only_selection_newcard_report.json --resume-dir $OUT_DIR/d4v2_emotion_only_selection_newcard_progress --max-new-tokens 256 --device cuda
08_d4v2_eprofile_selection|$PY scripts/run_response_inference.py --input $SEL_INPUT --model $MODEL_DIR --adapter $D4V2_ADAPTER --condition-jsonl $BOTH_COND_SEL --condition-fields both --output $OUT_DIR/d4v2_emotion_profile_selection_newcard.jsonl --performance-report $OUT_DIR/d4v2_emotion_profile_selection_newcard_report.json --resume-dir $OUT_DIR/d4v2_emotion_profile_selection_newcard_progress --max-new-tokens 256 --device cuda
09_m1_t4_arms_selection|$PY scripts/compare_response_arms.py --input $SEL_INPUT --arm t4_off=$OUT_DIR/t4_longweight_selection_newcard.jsonl --arm d4v2_off=$OUT_DIR/d4v2_selection_newcard.jsonl --main t4_off --base d4v2_off --output $OUT_DIR/t4_arms_comparison_newcard.json
10_m2_t4_val_errors|$PY scripts/analyze_response_errors.py --gold $VAL_INPUT --arm t4=$OUT_DIR/t4_longweight_val_newcard.jsonl --arm d4v2=$OUT_DIR/d4v2_val_newcard.jsonl --main t4 --base d4v2 --output reports/t4_val_error_analysis_newcard.json
11_m3a_d4v2_emotion_vs_off|$PY scripts/compare_response_arms.py --input $SEL_INPUT --arm off=$OUT_DIR/d4v2_selection_newcard.jsonl --arm emotion=$OUT_DIR/d4v2_emotion_only_selection_newcard.jsonl --arm eprofile=$OUT_DIR/d4v2_emotion_profile_selection_newcard.jsonl --main emotion --base off --output $OUT_DIR/t5_d4v2_emotion_vs_off_newcard.json
12_m3b_d4v2_eprofile_vs_off|$PY scripts/compare_response_arms.py --input $SEL_INPUT --arm off=$OUT_DIR/d4v2_selection_newcard.jsonl --arm emotion=$OUT_DIR/d4v2_emotion_only_selection_newcard.jsonl --arm eprofile=$OUT_DIR/d4v2_emotion_profile_selection_newcard.jsonl --main eprofile --base off --output $OUT_DIR/t5_d4v2_eprofile_vs_off_newcard.json
13_m3c_t4_emotion_vs_off|$PY scripts/compare_response_arms.py --input $SEL_INPUT --arm off=$OUT_DIR/t4_longweight_selection_newcard.jsonl --arm emotion=$OUT_DIR/t4_emotion_only_selection_newcard.jsonl --arm eprofile=$OUT_DIR/t4_emotion_profile_selection_newcard.jsonl --main emotion --base off --output $OUT_DIR/t5_t4_emotion_vs_off_newcard.json
14_m3d_t4_eprofile_vs_off|$PY scripts/compare_response_arms.py --input $SEL_INPUT --arm off=$OUT_DIR/t4_longweight_selection_newcard.jsonl --arm emotion=$OUT_DIR/t4_emotion_only_selection_newcard.jsonl --arm eprofile=$OUT_DIR/t4_emotion_profile_selection_newcard.jsonl --main eprofile --base off --output $OUT_DIR/t5_t4_eprofile_vs_off_newcard.json
15_t5_decide|$PY scripts/remote/decide_conditional_fullval.py --comparison $OUT_DIR/t5_d4v2_emotion_vs_off_newcard.json --comparison $OUT_DIR/t5_d4v2_eprofile_vs_off_newcard.json --comparison $OUT_DIR/t5_t4_emotion_vs_off_newcard.json --comparison $OUT_DIR/t5_t4_eprofile_vs_off_newcard.json --output $OUT_DIR/t5_conditional_decision.json
STAGES
}
