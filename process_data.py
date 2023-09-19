from datasets import load_from_disk
import jsonlines
import json

##### GSM8k
# test set
gsm8k_test_set = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/gsm8k_test_set_cache/')
gsm8k_test_set.to_json('data/gsm8k_test_set.json')
gsm8k_test_set = list(iter(jsonlines.open('data/gsm8k_test_set.json')))
with open('data/gsm8k_test_set.json','w') as f:
    json.dump(gsm8k_test_set, f, indent=2, ensure_ascii=False)

# Natural Language
gsm8k_nl = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/gsm8k_nl_cot_cache/')
gsm8k_nl.to_json('data/gsm8k_nl.json')
gsm8k_nl = list(iter(jsonlines.open('data/gsm8k_nl.json')))
with open('data/gsm8k_nl.json','w') as f:
    json.dump(gsm8k_nl, f, indent=2, ensure_ascii=False)

# Python
gsm8k_python_cdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/gsm8k_hybrid_cot_cache/')
gsm8k_python_cdp.to_json('data/gsm8k_python_cdp.json')
gsm8k_python_cdp = list(iter(jsonlines.open('data/gsm8k_python_cdp.json')))
with open('data/gsm8k_python_cdp.json','w') as f:
    json.dump(gsm8k_python_cdp, f, indent=2, ensure_ascii=False)


gsm8k_python_sdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/gsm8k_hybrid_baseline_cot_cache/')
gsm8k_python_sdp.to_json('data/gsm8k_python_sdp.json')
gsm8k_python_sdp = list(iter(jsonlines.open('data/gsm8k_python_sdp.json')))
with open('data/gsm8k_python_sdp.json','w') as f:
    json.dump(gsm8k_python_sdp, f, indent=2, ensure_ascii=False)


gsm8k_python_ndp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/gsm8k_hybrid_var_cot_cache/')
gsm8k_python_ndp.to_json('data/gsm8k_python_ndp.json')
gsm8k_python_ndp = list(iter(jsonlines.open('data/gsm8k_python_ndp.json')))
with open('data/gsm8k_python_ndp.json','w') as f:
    json.dump(gsm8k_python_ndp, f, indent=2, ensure_ascii=False)


# Wolfram
gsm8k_wolfram_cdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/gsm8k_hybrid_cot_cache/')
gsm8k_wolfram_cdp.to_json('data/gsm8k_wolfram_cdp.json')
gsm8k_wolfram_cdp = list(iter(jsonlines.open('data/gsm8k_wolfram_cdp.json')))
with open('data/gsm8k_wolfram_cdp.json','w') as f:
    json.dump(gsm8k_wolfram_cdp, f, indent=2, ensure_ascii=False)


gsm8k_wolfram_sdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/gsm8k_hybrid_baseline_cot_cache/')
gsm8k_wolfram_sdp.to_json('data/gsm8k_wolfram_sdp.json')
gsm8k_wolfram_sdp = list(iter(jsonlines.open('data/gsm8k_wolfram_sdp.json')))
with open('data/gsm8k_wolfram_sdp.json','w') as f:
    json.dump(gsm8k_wolfram_sdp, f, indent=2, ensure_ascii=False)


gsm8k_wolfram_ndp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/gsm8k_hybrid_var_cot_mathematica_cache/')
gsm8k_wolfram_ndp.to_json('data/gsm8k_wolfram_ndp.json')
gsm8k_wolfram_ndp = list(iter(jsonlines.open('data/gsm8k_wolfram_ndp.json')))
with open('data/gsm8k_wolfram_ndp.json','w') as f:
    json.dump(gsm8k_wolfram_ndp, f, indent=2, ensure_ascii=False)



##### MathQA
# test set
mathqa_test_set = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/mathqa_test_set_cache/')
mathqa_test_set.to_json('data/mathqa_test_set.json')
mathqa_test_set = list(iter(jsonlines.open('data/mathqa_test_set.json')))
with open('data/mathqa_test_set.json','w') as f:
    json.dump(mathqa_test_set, f, indent=2, ensure_ascii=False)

# Natural Language
mathqa_nl = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/mathqa_nl_cot_cache/')
mathqa_nl.to_json('data/mathqa_nl.json')
mathqa_nl = list(iter(jsonlines.open('data/mathqa_nl.json')))
with open('data/mathqa_nl.json','w') as f:
    json.dump(mathqa_nl, f, indent=2, ensure_ascii=False)

# Python
mathqa_python_cdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/mathqa_hybrid_cot_cache/')
mathqa_python_cdp.to_json('data/mathqa_python_cdp.json')
mathqa_python_cdp = list(iter(jsonlines.open('data/mathqa_python_cdp.json')))
with open('data/mathqa_python_cdp.json','w') as f:
    json.dump(mathqa_python_cdp, f, indent=2, ensure_ascii=False)


mathqa_python_sdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/mathqa_hybrid_baseline_cot_cache/')
mathqa_python_sdp.to_json('data/mathqa_python_sdp.json')
mathqa_python_sdp = list(iter(jsonlines.open('data/mathqa_python_sdp.json')))
with open('data/mathqa_python_sdp.json','w') as f:
    json.dump(mathqa_python_sdp, f, indent=2, ensure_ascii=False)


mathqa_python_ndp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/mathqa_hybrid_var_cot_cache/')
mathqa_python_ndp.to_json('data/mathqa_python_ndp.json')
mathqa_python_ndp = list(iter(jsonlines.open('data/mathqa_python_ndp.json')))
with open('data/mathqa_python_ndp.json','w') as f:
    json.dump(mathqa_python_ndp, f, indent=2, ensure_ascii=False)


# Wolfram
mathqa_wolfram_cdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/mathqa_hybrid_cot_cache/')
mathqa_wolfram_cdp.to_json('data/mathqa_wolfram_cdp.json')
mathqa_wolfram_cdp = list(iter(jsonlines.open('data/mathqa_wolfram_cdp.json')))
with open('data/mathqa_wolfram_cdp.json','w') as f:
    json.dump(mathqa_wolfram_cdp, f, indent=2, ensure_ascii=False)


mathqa_wolfram_sdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/mathqa_hybrid_baseline_cot_cache/')
mathqa_wolfram_sdp.to_json('data/mathqa_wolfram_sdp.json')
mathqa_wolfram_sdp = list(iter(jsonlines.open('data/mathqa_wolfram_sdp.json')))
with open('data/mathqa_wolfram_sdp.json','w') as f:
    json.dump(mathqa_wolfram_sdp, f, indent=2, ensure_ascii=False)


mathqa_wolfram_ndp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/mathqa_hybrid_var_cot_mathematica_cache/')
mathqa_wolfram_ndp.to_json('data/mathqa_wolfram_ndp.json')
mathqa_wolfram_ndp = list(iter(jsonlines.open('data/mathqa_wolfram_ndp.json')))
with open('data/mathqa_wolfram_ndp.json','w') as f:
    json.dump(mathqa_wolfram_ndp, f, indent=2, ensure_ascii=False)


##### SVAMP
# test set
svamp_test_set = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/svamp_test_set_cache/')
svamp_test_set.to_json('data/svamp_test_set.json')
svamp_test_set = list(iter(jsonlines.open('data/svamp_test_set.json')))
with open('data/svamp_test_set.json','w') as f:
    json.dump(svamp_test_set, f, indent=2, ensure_ascii=False)

# Natural Language
svamp_nl = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/svamp_nl_cot_cache/')
svamp_nl.to_json('data/svamp_nl.json')
svamp_nl = list(iter(jsonlines.open('data/svamp_nl.json')))
with open('data/svamp_nl.json','w') as f:
    json.dump(svamp_nl, f, indent=2, ensure_ascii=False)

# Python
svamp_python_cdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/svamp_hybrid_cot_cache/')
svamp_python_cdp.to_json('data/svamp_python_cdp.json')
svamp_python_cdp = list(iter(jsonlines.open('data/svamp_python_cdp.json')))
with open('data/svamp_python_cdp.json','w') as f:
    json.dump(svamp_python_cdp, f, indent=2, ensure_ascii=False)


svamp_python_sdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/svamp_hybrid_baseline_cot_cache/')
svamp_python_sdp.to_json('data/svamp_python_sdp.json')
svamp_python_sdp = list(iter(jsonlines.open('data/svamp_python_sdp.json')))
with open('data/svamp_python_sdp.json','w') as f:
    json.dump(svamp_python_sdp, f, indent=2, ensure_ascii=False)


svamp_python_ndp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/svamp_hybrid_var_cot_cache/')
svamp_python_ndp.to_json('data/svamp_python_ndp.json')
svamp_python_ndp = list(iter(jsonlines.open('data/svamp_python_ndp.json')))
with open('data/svamp_python_ndp.json','w') as f:
    json.dump(svamp_python_ndp, f, indent=2, ensure_ascii=False)


# Wolfram
svamp_wolfram_cdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/svamp_hybrid_cot_cache/')
svamp_wolfram_cdp.to_json('data/svamp_wolfram_cdp.json')
svamp_wolfram_cdp = list(iter(jsonlines.open('data/svamp_wolfram_cdp.json')))
with open('data/svamp_wolfram_cdp.json','w') as f:
    json.dump(svamp_wolfram_cdp, f, indent=2, ensure_ascii=False)


svamp_wolfram_sdp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data/svamp_hybrid_baseline_cot_cache/')
svamp_wolfram_sdp.to_json('data/svamp_wolfram_sdp.json')
svamp_wolfram_sdp = list(iter(jsonlines.open('data/svamp_wolfram_sdp.json')))
with open('data/svamp_wolfram_sdp.json','w') as f:
    json.dump(svamp_wolfram_sdp, f, indent=2, ensure_ascii=False)


svamp_wolfram_ndp = load_from_disk('/mnt/bn/trung-nas/data/paper_final/data_python/svamp_hybrid_var_cot_mathematica_cache/')
svamp_wolfram_ndp.to_json('data/svamp_wolfram_ndp.json')
svamp_wolfram_ndp = list(iter(jsonlines.open('data/svamp_wolfram_ndp.json')))
with open('data/svamp_wolfram_ndp.json','w') as f:
    json.dump(svamp_wolfram_ndp, f, indent=2, ensure_ascii=False)

