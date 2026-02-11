#!/usr/bin/env python3
"""
Transplant Inspire dexterous hands onto the base G1 robot USD.

Reads:
  - assets/g1_usd/g1.usd                              (base G1 body)
  - assets/g1_inspire_hand_usd/g1_inspire_flattened.usd (donor Inspire hands)

Writes:
  - assets/g1_usd/g1_inspire.usd                       (merged output)

Usage:
  python scripts/transplant_hands.py
"""

import os
import shutil
import re
from pxr import Sdf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE_USD = os.path.join(PROJECT_ROOT, "assets", "g1_usd", "g1.usd")
INSPIRE_USD = os.path.join(
    PROJECT_ROOT, "assets", "g1_inspire_hand_usd", "g1_inspire_flattened.usd"
)
OUTPUT_USD = os.path.join(PROJECT_ROOT, "assets", "g1_usd", "g1_inspire.usd")

# Root prim names
BASE_ROOT = "/g1_29dof_with_hand_rev_1_0"
INSPIRE_ROOT = "/g1_29dof_rev_1_0"

# ---------------------------------------------------------------------------
# Hand prim names
# ---------------------------------------------------------------------------

# Base G1 finger links (direct children of root) to REMOVE
BASE_FINGER_LINKS = [
    "left_hand_index_0_link",
    "left_hand_index_1_link",
    "left_hand_middle_0_link",
    "left_hand_middle_1_link",
    "left_hand_thumb_0_link",
    "left_hand_thumb_1_link",
    "left_hand_thumb_2_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
]

# Base G1 finger joints to REMOVE
BASE_FINGER_JOINTS = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

# Inspire finger links (direct children of root) to COPY
INSPIRE_FINGER_LINKS = [
    "L_index_proximal",
    "L_index_intermediate",
    "L_middle_proximal",
    "L_middle_intermediate",
    "L_pinky_proximal",
    "L_pinky_intermediate",
    "L_ring_proximal",
    "L_ring_intermediate",
    "L_thumb_proximal_base",
    "L_thumb_proximal",
    "L_thumb_intermediate",
    "L_thumb_distal",
    "R_index_proximal",
    "R_index_intermediate",
    "R_middle_proximal",
    "R_middle_intermediate",
    "R_pinky_proximal",
    "R_pinky_intermediate",
    "R_ring_proximal",
    "R_ring_intermediate",
    "R_thumb_proximal_base",
    "R_thumb_proximal",
    "R_thumb_intermediate",
    "R_thumb_distal",
]

# Inspire finger joints to COPY
INSPIRE_FINGER_JOINTS = [
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]

# Inspire prototype IDs used by hand prims (visuals + collisions)
INSPIRE_HAND_PROTO_IDS = [
    1, 2, 3, 5, 6, 15, 17, 18, 20, 22, 24, 25, 27, 28, 29, 33, 36, 39, 40,
    41, 43, 44, 45, 50, 51, 52, 56, 58, 59, 62, 66, 68, 70, 72, 74, 75, 76,
    78, 80, 81, 82, 83, 84, 86, 93, 95, 96, 97, 98, 104, 105, 106,
]

# Materials that exist in inspire but not base (need to copy)
MISSING_MATERIALS = ["material_white", "material_100100100"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_proto_refs(layer, prim_path):
    """Recursively collect all prototype reference paths from a prim tree."""
    refs = set()
    prim = layer.GetPrimAtPath(prim_path)
    if not prim:
        return refs
    for ref in prim.referenceList.GetAddedOrExplicitItems():
        if ref.assetPath == "" and "Prototype" in str(ref.primPath):
            refs.add(str(ref.primPath))
    for child in prim.nameChildren:
        refs.update(collect_proto_refs(layer, prim_path.AppendChild(child.name)))
    return refs


def remap_paths_recursive(layer, prim_path, remap_dict):
    """Recursively remap internal references and relationship targets."""
    prim = layer.GetPrimAtPath(prim_path)
    if not prim:
        return

    # Remap internal references (prototype refs)
    ref_items = prim.referenceList.GetAddedOrExplicitItems()
    new_refs = []
    changed = False
    for ref in ref_items:
        if ref.assetPath == "":
            old_path = str(ref.primPath)
            new_path = remap_dict.get(old_path)
            if new_path:
                new_refs.append(
                    Sdf.Reference(primPath=Sdf.Path(new_path))
                )
                changed = True
            else:
                new_refs.append(ref)
        else:
            new_refs.append(ref)
    if changed:
        prim.referenceList.ClearEdits()
        prim.referenceList.explicitItems = new_refs

    # Remap relationship targets
    for rel_name in list(prim.relationships.keys()):
        rel = prim.relationships[rel_name]
        targets = rel.targetPathList.GetAddedOrExplicitItems()
        new_targets = []
        rel_changed = False
        for t in targets:
            ts = str(t)
            remapped = None
            for old, new in remap_dict.items():
                if ts.startswith(old):
                    remapped = new + ts[len(old):]
                    break
            if remapped:
                new_targets.append(Sdf.Path(remapped))
                rel_changed = True
            else:
                new_targets.append(t)
        if rel_changed:
            rel.targetPathList.ClearEdits()
            rel.targetPathList.explicitItems = new_targets

    # Recurse into children
    for child in prim.nameChildren:
        remap_paths_recursive(
            layer, prim_path.AppendChild(child.name), remap_dict
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Base USD:    {BASE_USD}")
    print(f"Inspire USD: {INSPIRE_USD}")
    print(f"Output USD:  {OUTPUT_USD}")

    # ------------------------------------------------------------------
    # Step 1: Byte-copy base → output
    # ------------------------------------------------------------------
    print("\n[1] Copying base USD to output...")
    shutil.copy2(BASE_USD, OUTPUT_USD)

    # ------------------------------------------------------------------
    # Step 2: Open layers
    # ------------------------------------------------------------------
    print("[2] Opening layers...")
    src = Sdf.Layer.FindOrOpen(INSPIRE_USD)
    dst = Sdf.Layer.FindOrOpen(OUTPUT_USD)
    assert src, f"Failed to open {INSPIRE_USD}"
    assert dst, f"Failed to open {OUTPUT_USD}"

    dst_root = Sdf.Path(BASE_ROOT)
    src_root = Sdf.Path(INSPIRE_ROOT)

    # ------------------------------------------------------------------
    # Step 3: Remove old hand prims from dst
    # ------------------------------------------------------------------
    print("[3] Removing old hand prims...")

    # 3a. Remove old finger link prims
    for name in BASE_FINGER_LINKS:
        _remove_prim(dst, dst_root, name)

    # 3b. Remove old finger joints
    joints_path = dst_root.AppendChild("joints")
    for name in BASE_FINGER_JOINTS:
        _remove_prim(dst, joints_path, name)

    # 3c. Remove old wrist_yaw_link children
    for side in ["left", "right"]:
        wrist_path = dst_root.AppendChild(f"{side}_wrist_yaw_link")
        wrist = dst.GetPrimAtPath(wrist_path)
        if wrist:
            children_to_remove = [c.name for c in wrist.nameChildren]
            for cname in children_to_remove:
                _remove_prim(dst, wrist_path, cname)

    print(f"  Removed {len(BASE_FINGER_LINKS)} finger links, "
          f"{len(BASE_FINGER_JOINTS)} finger joints, wrist children")

    # ------------------------------------------------------------------
    # Step 4: Copy inspire hand content
    # ------------------------------------------------------------------
    print("[4] Copying inspire hand prims...")

    # 4a. Copy wrist_yaw_link children from inspire
    for side in ["left", "right"]:
        src_wrist = src_root.AppendChild(f"{side}_wrist_yaw_link")
        dst_wrist = dst_root.AppendChild(f"{side}_wrist_yaw_link")
        src_wrist_prim = src.GetPrimAtPath(src_wrist)
        for child in src_wrist_prim.nameChildren:
            src_child = src_wrist.AppendChild(child.name)
            dst_child = dst_wrist.AppendChild(child.name)
            ok = Sdf.CopySpec(src, src_child, dst, dst_child)
            assert ok, f"Failed to copy {src_child} → {dst_child}"

    # 4b. Copy finger link prims
    for name in INSPIRE_FINGER_LINKS:
        src_path = src_root.AppendChild(name)
        dst_path = dst_root.AppendChild(name)
        ok = Sdf.CopySpec(src, src_path, dst, dst_path)
        assert ok, f"Failed to copy finger link {name}"

    # 4c. Copy finger joints
    src_joints = src_root.AppendChild("joints")
    dst_joints = dst_root.AppendChild("joints")
    for name in INSPIRE_FINGER_JOINTS:
        src_path = src_joints.AppendChild(name)
        dst_path = dst_joints.AppendChild(name)
        ok = Sdf.CopySpec(src, src_path, dst, dst_path)
        assert ok, f"Failed to copy finger joint {name}"

    print(f"  Copied wrist children, {len(INSPIRE_FINGER_LINKS)} finger links, "
          f"{len(INSPIRE_FINGER_JOINTS)} finger joints")

    # ------------------------------------------------------------------
    # Step 5: Copy hand prototypes with renumbering
    # ------------------------------------------------------------------
    print("[5] Copying hand prototypes...")

    # Build remap: inspire proto path → new proto path in dst
    proto_remap = {}
    next_id = 48  # base has prototypes 1..47
    for inspire_id in INSPIRE_HAND_PROTO_IDS:
        old_path = f"/Flattened_Prototype_{inspire_id}"
        new_path = f"/Flattened_Prototype_{next_id}"
        proto_remap[old_path] = new_path

        ok = Sdf.CopySpec(src, Sdf.Path(old_path), dst, Sdf.Path(new_path))
        assert ok, f"Failed to copy prototype {old_path} → {new_path}"
        next_id += 1

    print(f"  Copied {len(proto_remap)} prototypes (IDs 48..{next_id - 1})")

    # ------------------------------------------------------------------
    # Step 6: Copy missing materials
    # ------------------------------------------------------------------
    print("[6] Copying missing materials...")
    src_looks = src_root.AppendChild("Looks")
    dst_looks = dst_root.AppendChild("Looks")
    for mat_name in MISSING_MATERIALS:
        src_mat = src_looks.AppendChild(mat_name)
        dst_mat = dst_looks.AppendChild(mat_name)
        if src.GetPrimAtPath(src_mat):
            ok = Sdf.CopySpec(src, src_mat, dst, dst_mat)
            assert ok, f"Failed to copy material {mat_name}"
            print(f"  Copied material: {mat_name}")

    # ------------------------------------------------------------------
    # Step 7: Fix all path references
    # ------------------------------------------------------------------
    print("[7] Fixing path references...")

    # Build full remap dictionary
    full_remap = dict(proto_remap)
    # Also remap inspire root → dst root (for relationship targets, material bindings)
    full_remap[INSPIRE_ROOT] = BASE_ROOT

    # 7a. Fix refs in copied prototypes
    for new_path in proto_remap.values():
        remap_paths_recursive(dst, Sdf.Path(new_path), full_remap)

    # 7b. Fix refs in copied wrist children
    for side in ["left", "right"]:
        wrist_path = dst_root.AppendChild(f"{side}_wrist_yaw_link")
        wrist = dst.GetPrimAtPath(wrist_path)
        for child in wrist.nameChildren:
            child_path = wrist_path.AppendChild(child.name)
            remap_paths_recursive(dst, child_path, full_remap)

    # 7c. Fix refs in copied finger links
    for name in INSPIRE_FINGER_LINKS:
        remap_paths_recursive(dst, dst_root.AppendChild(name), full_remap)

    # 7d. Fix refs in copied finger joints
    for name in INSPIRE_FINGER_JOINTS:
        remap_paths_recursive(dst, dst_joints.AppendChild(name), full_remap)

    print("  All references remapped")

    # ------------------------------------------------------------------
    # Step 8: Save
    # ------------------------------------------------------------------
    print("[8] Saving...")
    dst.Save()
    print(f"  Saved: {OUTPUT_USD}")

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("\n[Verification]")
    verify = Sdf.Layer.FindOrOpen(OUTPUT_USD)
    root = verify.GetPrimAtPath(Sdf.Path(BASE_ROOT))
    assert root, "Root prim missing!"
    print(f"  Root prim: {BASE_ROOT} (type={root.typeName})")

    # Check finger links exist
    for name in INSPIRE_FINGER_LINKS:
        p = verify.GetPrimAtPath(Sdf.Path(BASE_ROOT).AppendChild(name))
        assert p, f"Missing finger link: {name}"
    print(f"  All {len(INSPIRE_FINGER_LINKS)} finger links present")

    # Check finger joints exist with valid targets
    joints = Sdf.Path(BASE_ROOT).AppendChild("joints")
    for name in INSPIRE_FINGER_JOINTS:
        jp = verify.GetPrimAtPath(joints.AppendChild(name))
        assert jp, f"Missing finger joint: {name}"
        for rn in ["physics:body0", "physics:body1"]:
            rel = jp.relationships.get(rn)
            assert rel, f"Joint {name} missing {rn}"
            targets = rel.targetPathList.GetAddedOrExplicitItems()
            for t in targets:
                assert str(t).startswith(BASE_ROOT), (
                    f"Joint {name} {rn} has wrong root: {t}"
                )
    print(f"  All {len(INSPIRE_FINGER_JOINTS)} finger joints valid")

    # Check prototypes referenced by hand prims resolve
    for name in INSPIRE_FINGER_LINKS[:2]:
        refs = collect_proto_refs(
            verify, Sdf.Path(BASE_ROOT).AppendChild(name)
        )
        for ref_path in refs:
            assert verify.GetPrimAtPath(Sdf.Path(ref_path)), (
                f"Unresolved prototype: {ref_path}"
            )
    print("  Prototype references resolve")

    # Check materials
    looks = Sdf.Path(BASE_ROOT).AppendChild("Looks")
    for mat in MISSING_MATERIALS:
        assert verify.GetPrimAtPath(looks.AppendChild(mat)), (
            f"Missing material: {mat}"
        )
    print("  Materials present")

    # Count total joints
    joints_prim = verify.GetPrimAtPath(joints)
    n_joints = len(list(joints_prim.nameChildren))
    print(f"  Total joints: {n_joints} (expected 53)")

    # Count root children
    n_children = len(list(root.nameChildren))
    print(f"  Root children: {n_children}")

    print("\nDone!")


def _remove_prim(layer, parent_path, child_name):
    """Remove a child prim spec from a parent in an Sdf layer."""
    child_path = parent_path.AppendChild(child_name)
    if not layer.GetPrimAtPath(child_path):
        return False
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(child_path, Sdf.Path.emptyPath)
    if not layer.Apply(edit):
        raise RuntimeError(f"Failed to remove {child_path}")
    return True


if __name__ == "__main__":
    main()
