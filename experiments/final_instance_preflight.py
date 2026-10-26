#!/usr/bin/env python3
"""Real-IAD metadata-only preflight: run before expensive frozen final GPU test."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"experiments"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import final_decision_core as core

def validate(dataset_root,json_dir):
    import vmb_visionad_new as data
    facts=[]
    for cl in core.CLASSES:
        index=data.RealIADJSONIndex(
            root=dataset_root,json_dir=json_dir,only_classes=[cl])
        train=[s for s in index.train_samples if s.cls_name==cl]
        test=[s for s in index.test_samples if s.cls_name==cl]
        by=core.group_samples(test)
        if len(by)<(core.A_PROBE_GROUPS+core.B_PROBE_GROUPS+
                    3*core.GROUPS_PER_PHASE):
            raise RuntimeError(f"Not enough physical groups for {cl}")
        for seed in core.SEEDS:
            support=core.choose_supports(train,seed)
            probes,phases,groups=core.make_split(test,support,seed)
            flat=sum(phases,[])
            query=core.query_tape(phases)
            labels=[int(not flat[t-1].is_good) for t in sorted(query)]
            for mod in ("A","B"):
                label_objects={}
                for s in probes[mod]:
                    id=core.object_id(s.path)
                    label_objects[id]=max(label_objects.get(id,0),int(not s.is_good))
                if set(label_objects.values())!={0,1}:
                    raise RuntimeError(f"{cl}/s{seed} probe {mod} no normal+anomaly groups")
            facts.append({
                "class":cl,"seed":seed,"train_normal_object_groups":
                    len(core.group_samples(s for s in train if s.is_good)),
                "test_object_groups":len(by),
                "support_image_count":len(support),
                "support_unique_instances":len(set(map(core.object_id,support))),
                "group_counts":[len(groups["A"]),len(groups["B"])] +
                               [len(z) for z in groups["phases"]],
                "stream_images":len(flat),"queried_images":len(query),
                "query_rate_actual":len(query)/len(flat),
                "queried_verified_normal":labels.count(0),
                "queried_defect":labels.count(1),
                "query_groups_unique":len({
                    core.object_id(flat[t-1].path) for t in query})==len(query),
                "manifest_digest":core.digest(groups),
            })
    print(json.dumps({"complete":True,"stage":"metadata_preflight",
                      "conditions":facts},indent=2,ensure_ascii=False))
    return facts

if __name__=="__main__":
    root=sys.argv[1] if len(sys.argv)>1 else "G:/dataset/realiad"
    json_dir=sys.argv[2] if len(sys.argv)>2 else str(Path(root)/"jsons"/"realiad_jsons")
    validate(root,json_dir)
