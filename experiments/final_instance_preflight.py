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

def verify_native_memory_ops():
    """Actual v22 CPU operators; no dataset images or GPU required."""
    import copy
    import torch
    import test_phase3_AIF_simplified_v22 as impl
    import tmra_compact_memory as mm
    import final_decision_core as core
    bank=impl.PatchVMB(
        device="cpu",K_normal_fixed=32,K_normal_dyn=320,
        K_normal_ltm=64,K_defect=4,K_concept=8,knn_k_normal=5,
        corr_cfg=impl.CorrectionConfig(enabled=False),
        use_defect_bank=False)
    bank.bank_dtype=torch.float16
    reference=torch.zeros((6,4096),dtype=torch.float32)
    reference[torch.arange(6),torch.arange(6)]=1.0
    bank.build_fixed_normal_from_support(
        "p0",[[reference]],mode=0,view="id",
        normal_seed_patches=6,seed=0)
    setting=core.MemorySettings()
    arms={name:mm.Arm(name,copy.deepcopy(bank),setting,seed=37)
          for name in ("greedy_coreset","hierarchy")}
    reserved={}
    for name,arm in arms.items():
        stm,ltm,buf=mm.codebooks(arm.bank)
        nominal=0
        for cb in stm+ltm:
            nominal+=cb.K*cb.feat_dim*torch.empty(
                (),dtype=cb.dtype).element_size()+80*cb.K
        for cb in buf:
            nominal+=cb.K*cb.feat_dim*4+80*cb.K
        reserved[name]=nominal
        if nominal>core.BYTE_CEILING:
            raise AssertionError("Nominal byte ceiling exceeded")
    vals=list(reserved.values())
    if abs(vals[0]-vals[1])/max(vals)>.05:
        raise AssertionError(f"Native memory byte-reservation mismatch: {reserved}")
    torch.manual_seed(100)
    for step in range(1,4):
        f=torch.nn.functional.normalize(
            torch.randn((24,4096),dtype=torch.float32),dim=1)
        for arm in arms.values():
            arm.update(impl,[f],{0:list(range(24))},step)
            if arm.audit()["feedback_events"]!=step:
                raise AssertionError("CPU synthetic feedback mismatch")
    print("[FINAL] Real v22 two-arm CPU operator + byte-reservation PASS",reserved,flush=True)

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
    verify_native_memory_ops()
    print(json.dumps({"complete":True,"stage":"metadata_plus_native_CPU_preflight",
                      "conditions":facts},indent=2,ensure_ascii=False))
    return facts

if __name__=="__main__":
    root=sys.argv[1] if len(sys.argv)>1 else "G:/dataset/realiad"
    json_dir=sys.argv[2] if len(sys.argv)>2 else str(Path(root)/"jsons"/"realiad_jsons")
    validate(root,json_dir)
