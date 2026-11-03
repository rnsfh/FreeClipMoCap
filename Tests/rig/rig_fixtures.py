_MIXAMO_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def mixamo_parents(prefix: str = "mixamorig:") -> dict[str, str | None]:
    parents: dict[str, str | None] = {"Hips": None, "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1"}
    parents.update({"Neck": "Spine2", "Head": "Neck", "HeadTop_End": "Head", "headfront": "Head"})
    for side in ("Left", "Right"):
        parents.update({
            f"{side}Shoulder": "Spine2",
            f"{side}Arm": f"{side}Shoulder",
            f"{side}ForeArm": f"{side}Arm",
            f"{side}Hand": f"{side}ForeArm",
            f"{side}UpLeg": "Hips",
            f"{side}Leg": f"{side}UpLeg",
            f"{side}Foot": f"{side}Leg",
            f"{side}ToeBase": f"{side}Foot",
            f"{side}Toe_End": f"{side}ToeBase",
        })
        for finger in _MIXAMO_FINGERS:
            parent = f"{side}Hand"
            for joint in range(1, 5):
                parents[f"{side}Hand{finger}{joint}"] = parent
                parent = f"{side}Hand{finger}{joint}"
    parents.update({"Bone_007": "Hips", "Bone_006": "Bone_007"})
    return {
        (name if name in ("headfront", "Bone_007", "Bone_006") else f"{prefix}{name}"):
        (None if parent is None else parent if parent in ("headfront", "Bone_007", "Bone_006") else f"{prefix}{parent}")
        for name, parent in parents.items()
    }


def unreal_style_parents() -> dict[str, str | None]:
    parents: dict[str, str | None] = {"root": None, "pelvis": "root"}
    parents.update({"spine_01": "pelvis", "spine_02": "spine_01", "spine_03": "spine_02", "spine_04": "spine_03"})
    parents.update({"spine_05": "spine_04", "neck_01": "spine_05", "neck_02": "neck_01", "head": "neck_02"})
    for side in ("l", "r"):
        parents.update({
            f"clavicle_{side}": "spine_05",
            f"upperarm_{side}": f"clavicle_{side}",
            f"upperarm_twist_01_{side}": f"upperarm_{side}",
            f"lowerarm_{side}": f"upperarm_{side}",
            f"hand_{side}": f"lowerarm_{side}",
            f"thigh_{side}": "pelvis",
            f"calf_{side}": f"thigh_{side}",
            f"foot_{side}": f"calf_{side}",
            f"ball_{side}": f"foot_{side}",
            f"ik_foot_{side}": "root",
        })
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            parent = f"hand_{side}"
            if finger != "thumb":
                parents[f"{finger}_metacarpal_{side}"] = parent
                parent = f"{finger}_metacarpal_{side}"
            for joint in range(1, 4):
                parents[f"{finger}_0{joint}_{side}"] = parent
                parent = f"{finger}_0{joint}_{side}"
    return parents


def metarig_style_parents() -> dict[str, str | None]:
    parents: dict[str, str | None] = {"spine": None, "spine.001": "spine", "spine.002": "spine.001"}
    parents.update({"spine.003": "spine.002", "neck": "spine.003", "head": "neck"})
    for side in ("L", "R"):
        parents.update({
            f"shoulder.{side}": "spine.003",
            f"upper_arm.{side}": f"shoulder.{side}",
            f"forearm.{side}": f"upper_arm.{side}",
            f"hand.{side}": f"forearm.{side}",
            f"thigh.{side}": "spine",
            f"shin.{side}": f"thigh.{side}",
            f"foot.{side}": f"shin.{side}",
            f"toe.{side}": f"foot.{side}",
        })
    return parents
