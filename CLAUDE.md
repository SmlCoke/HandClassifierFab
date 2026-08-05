# AGENTS.md

## I. Purpose

This file defines the repository-wide instructions for coding agents.

Agents must follow these instructions when inspecting, modifying, testing, or
documenting this repository.

The primary goals are:

1. Preserve repository correctness and maintainability.
2. Make only changes required by the current task.
3. Avoid destructive, unrelated, or speculative modifications.

---

## II. Instruction Priority

Follow instructions in this order:

1. Explicit instructions in the current user request.
2. This repository-level `AGENTS.md`.
3. Existing project conventions inferred from nearby code.

Higher-priority instructions override lower-priority instructions.

Do not interpret a previous task's temporary authorization as authorization for
the current task.

When two applicable instructions conflict and the conflict cannot be resolved
safely, stop before modifying files and explain the conflict.

---

## III. Repository Overview

### 3.1 Project purpose

We are developing a real-time sign language recognition system designed for low-compute (0.8 TOPS) edge devices. The system comprises a pipeline of three models operating in series:
1. Palm Detector: Processes images captured by the camera and outputs palm bounding box coordinates along with the coordinates of two auxiliary points.
2. Hand Landmarker: Performs inference on the Hand ROI defined by the palm bounding box to determine the coordinates of 21 skeletal keypoints, as well as hand presence and handedness (left/right) confidence scores.
3. Gloss Translator: An isolated sign classification model that utilizes the outputs from the Palm Detector and Hand Landmarker; it maintains a temporal window of a specific duration and performs temporal modeling to output the sign language gloss corresponding to the action performed during that interval.

本仓库的目的是实现一个左/右手二分类的小模型的定义、训练、评估的完整系统，用于在 Iris 系列模型所需数据集的自动化标注链路中发挥作用。

### 3.2 Entry-point documents

- `docs\HCF_system\HCF_annotating_workflow.md`: Referred to as the "workflow" document; it explains the current system's workflow and procedures rather than serving merely as an operational manual. 
- `docs\HCF_system\HCF_quick_start.md`: Referred to as the "quick_start" document; a simplified version of the "workflow" document containing instructions for executing the full process, designed for getting started quickly.

## IV. General Working Rules

### 4.1 Environment and General Rules

1. The system must be capable of running on both Windows and Linux.
2. Use `make` and `Makefile` to build the project.
3. Use Git for version control.
4. Use `requirements.txt` to configure environment dependencies and `conda` to manage Python virtual environments.

### 4.2 Docs Modifying Rules

- The "workflow" document records only the commands, content, and underlying principles for each operational step of the system; it is independent of the system's historical state, the training status of models on the server, the state of the server-side data warehouse, and the project's future plans. It is necessary to explain the command and input (including directory locations) for each step, the actions performed, the output (including directory locations), and the rationale behind parameter adjustments in the YAML configuration file. Please keep this principle in mind when making modifications.
- The "quick_start" document is a simplified version of the "workflow" document; it contains only the commands for each operational step and omits explanations of the underlying principles. Include the name of the process stage for each step and briefly describe the inputs and outputs. Please keep this principle in mind when making modifications. 
- The "current_status" document records the current state of the system and the server-side data warehouse. Please keep this principle in mind when making modifications.


These three documents have distinct roles and independent content; each should avoid extensive discussion of the material covered in the others.
These three documents serve as the primary interface documentation for the current repository and are critical; they must be kept synchronized whenever there are subsequent updates to code, configurations, or other documentation.

### 4.3 Principle of simplification

1. Whether during manual operations or automated execution of the repository's programs, performing a hash check (SHA256) at every step is prohibited, as this results in significant waste of time and excessive disk space usage.

### 4.4 Principle of Environment Dependency

- Whenever you modify the code or add a new model, remember to check if the repository's environment dependencies have changed. If they have, update `requirements.txt` and remind me to update the `anfab` environment.