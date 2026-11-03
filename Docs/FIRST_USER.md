# First user: LifeInArena

FreeClipMocap is built first to make animations for **LifeInArena** (a separate repository),
a Godot 4.7.1 .NET game. This file holds what FreeClipMocap needs to know about it. LifeInArena's own docs and
Claude memory are the source of truth for the game.

## Hero rigs

- Bodies are rigged in Meshy: armature `SmartRigArmature`, **66 bones named `mixamorig:*`** (5 fingers × 3 joints),
  T-pose, in **metres**. Some races have extra chains (tails, ears, manes, snouts named `Bone_0xx`, `headfront`)
  that FreeClipMocap should leave alone.
- Some rigs had missing or poor finger bones that the user fixed by hand; check finger bones exist before solving
  fingers.
- Races vary a lot in proportion (Thistlekin ~1.05–1.10 m, dwarves ~1.45 m, gnolls ~2 m, elves ~1.9 m), so
  clips should be checked on several bodies.
- Hero `.blend` files live in `LifeInArena/src/Models/Characters/Heroes/<Race><Sex>/`.
- Checked 2026-11-03 (all 12 heroes, read-only): every rig has the full Mixamo set of 65 `mixamorig:*` bones
  (fingers 4 bones each, `…4` being the tip), quaternion rotation, a single root `mixamorig:Hips`. Only
  `skarn_female` lacks the two `Toe_End` tip bones. Extras: `headfront` under the Head (all but the humans), gnoll
  tails `Bone_0xx` (3–4 bones) under the Hips. The armature object is `SmartRigArmature`, except the humans'
  `Armature`, so nothing may rely on the object's name.

## Current animation pipeline (Mixamo clips)

- The user picks clips on Mixamo; FBX Binary, Without Skin, 30 fps, In Place for locomotion. Saved in
  `LifeInArena/src/Animations/Mixamo/`.
- Godot import: Mixamo BoneMap onto the humanoid profile; `fix_silhouette/filter` = Hips, Spine, Chest,
  UpperChest, Neck, Head on every hero `.import`. `root_scale` 100 for Mixamo FBX (centimetres), 1 for Meshy
  (metres).
- Clips are registered in `src/Scripts/Characters/HeroClips.cs`. The game strips horizontal hip travel from one-shot
  clips (the game moves the hero).
- The user rejected a clip whose fingers were missing (fingers stayed in rest pose), so finger data matters.

## What FreeClipMocap must produce for it

- An FBX animation on a `mixamorig:*` skeleton that drops into the same import settings as a Mixamo clip.
- In-place option for locomotion; clean loops for idles.
- Full finger animation, or a deliberate finger pose, never fingers stuck in rest pose.
