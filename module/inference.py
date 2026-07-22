"""Alpamayo inference utilities."""

import copy
import math
import os
import re

import torch
import numpy as np

from transformers import BitsAndBytesConfig

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper

from . import config as cfg
from .alpamayo_compat import patch_legacy_hydra_targets
from .vlm_generate_optimization import VlmGenerateTiming, optimized_vlm_generate

patch_legacy_hydra_targets()

SUPPORTED_CUDA_LINALG_LIBRARIES = {"default", "cusolver", "magma"}
ALPAMAYO_MODEL_NAME = "nvidia/Alpamayo-1.5-10B"
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>|</s>|<s>")
VQA_ANSWER_TERMINATORS = (
    "<|answer_end|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|cot_start|>",
    "<|cot_end|>",
    "<|meta_action_start|>",
    "<|meta_action_end|>",
    "<|question_start|>",
    "<|question_end|>",
)


def require_cuda_runtime():
    """Fail before model loading when PyTorch cannot use an NVIDIA GPU."""

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable to PyTorch, and Alpamayo cannot run on CPU. "
            "Verify that this process is inside a GPU allocation, that `nvidia-smi` "
            "can access the assigned GPU, and that another process (such as CARLA) "
            "does not already own an Exclusive_Process GPU. "
            f"CUDA_VISIBLE_DEVICES={visible_devices}."
        )

    try:
        probe = torch.empty(1, device="cuda")
        del probe
    except Exception as exc:
        raise RuntimeError(
            "PyTorch detected CUDA but could not create a CUDA tensor. Check the "
            "NVIDIA driver and whether another process owns an Exclusive_Process GPU. "
            f"CUDA_VISIBLE_DEVICES={visible_devices}."
        ) from exc

    return torch.cuda.get_device_name(0)


def configure_cuda_linalg_library(library: str | None):
    """Set PyTorch's preferred CUDA linalg backend when supported."""

    if library is None:
        return None

    normalized = library.strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized not in SUPPORTED_CUDA_LINALG_LIBRARIES:
        supported = ", ".join(sorted(SUPPORTED_CUDA_LINALG_LIBRARIES))
        raise ValueError(
            f"Unsupported CUDA linalg library '{library}'. Expected one of: {supported}."
        )
    if not torch.cuda.is_available():
        return None

    preferred_linalg_library = getattr(torch.backends.cuda, "preferred_linalg_library", None)
    if preferred_linalg_library is None:
        return None
    return preferred_linalg_library(normalized)


def load_model(use_quantization: bool, device_map="auto"):
    """Load Alpamayo model and processor."""
    require_cuda_runtime()
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = Alpamayo1_5.from_pretrained(
            ALPAMAYO_MODEL_NAME,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    else:
        if device_map:
            model = Alpamayo1_5.from_pretrained(
                ALPAMAYO_MODEL_NAME,
                dtype=torch.bfloat16,
                device_map=device_map,
            )
        else:
            model = Alpamayo1_5.from_pretrained(
                ALPAMAYO_MODEL_NAME,
                dtype=torch.bfloat16,
            ).to("cuda")

    processor = helper.get_processor(model.tokenizer)
    return model, processor


def _prime_oom_pipeline(model):
    """Reset the OOM-free demand-layering pipeline before an inference, if any.

    When the model was loaded with ``module.oom_offload.load_offloaded_model``
    (``--oom-free``), a ``TriHookPipeline`` is attached as ``model._oom_pipeline``
    and must be primed once per inference. A normally-loaded model has no such
    attribute and this is a no-op.
    """
    pipeline = getattr(model, "_oom_pipeline", None)
    if pipeline is not None:
        pipeline.start_iteration()


def _configured_camera_ids():
    camera_ids = tuple(int(spec["alpamayo_id"]) for spec in cfg.CAMERA_SPECS)
    camera_names = tuple(str(spec["name"]) for spec in cfg.CAMERA_SPECS)
    expected_names = (
        "cam_front_left",
        "cam_front_wide",
        "cam_front_right",
        "cam_front_tele",
    )
    if camera_ids != (0, 1, 2, 6):
        raise ValueError(
            "CAMERA_SPECS must be ordered as Alpamayo cameras [0, 1, 2, 6]"
        )
    if camera_names != expected_names:
        raise ValueError(
            "CAMERA_SPECS must be ordered as front-left, front-wide, "
            "front-right, front-tele"
        )
    return camera_ids


def prepare_model_input(
    images_array,
    history_xyz,
    history_rot,
    *,
    camera_indices=None,
):
    """Validate and convert synchronized CARLA data to Alpamayo tensors."""

    images_array = np.asarray(images_array)
    history_xyz = np.asarray(history_xyz)
    history_rot = np.asarray(history_rot)
    configured_camera_ids = _configured_camera_ids()
    if camera_indices is None:
        camera_ids = configured_camera_ids
    else:
        if isinstance(camera_indices, torch.Tensor):
            camera_indices = camera_indices.detach().cpu().tolist()
        camera_ids = tuple(int(camera_id) for camera_id in camera_indices)
        if camera_ids != configured_camera_ids:
            raise ValueError(
                "camera_indices must match configured Alpamayo order "
                f"{list(configured_camera_ids)}; got {list(camera_ids)}"
            )

    if images_array.ndim != 5 or images_array.shape[:2] != (
        len(camera_ids),
        cfg.NUM_FRAMES,
    ):
        raise ValueError(
            "images_array must have shape "
            f"({len(camera_ids)}, {cfg.NUM_FRAMES}, H, W, C); got {images_array.shape}"
        )
    if images_array.shape[-1] != cfg.IMG_CHANNELS:
        raise ValueError(
            f"images_array must have {cfg.IMG_CHANNELS} channels; got {images_array.shape[-1]}"
        )
    if images_array.dtype != np.uint8:
        raise TypeError(f"images_array must use uint8 pixels; got {images_array.dtype}")
    if history_xyz.shape != (cfg.NUM_HISTORY, 3):
        raise ValueError(
            f"history_xyz must have shape ({cfg.NUM_HISTORY}, 3); got {history_xyz.shape}"
        )
    if history_rot.shape != (cfg.NUM_HISTORY, 3, 3):
        raise ValueError(
            "history_rot must have shape "
            f"({cfg.NUM_HISTORY}, 3, 3); got {history_rot.shape}"
        )
    if not np.isfinite(history_xyz).all() or not np.isfinite(history_rot).all():
        raise ValueError("ego history must contain only finite values")

    images = torch.from_numpy(images_array).permute(0, 1, 4, 2, 3).contiguous()
    hist_xyz = torch.from_numpy(history_xyz).float().unsqueeze(0).unsqueeze(0)
    hist_rot = torch.from_numpy(history_rot).float().unsqueeze(0).unsqueeze(0)
    return {
        "image_frames": images,
        "camera_indices": torch.tensor(camera_ids, dtype=torch.long),
        "ego_history_xyz": hist_xyz,
        "ego_history_rot": hist_rot,
    }


def run_inference(
    model,
    processor,
    data,
    navigation_text: str | None = None,
    navigation_weight: float = 1.0,
    vlm_generate_timing: VlmGenerateTiming | None = None,
    disable_unused_generate_logits: bool = True,
    vlm_image_pixels: int | None = None,
):
    """Run Alpamayo inference locally.

    ``navigation_text`` conditions the trajectory prompt. ``navigation_weight``
    uses Alpamayo's CFG navigation path when it differs from 1.0.
    """

    nav_text = navigation_text.strip() if isinstance(navigation_text, str) else ""
    if not math.isfinite(float(navigation_weight)) or float(navigation_weight) < 0:
        raise ValueError("navigation_weight must be a non-negative finite number")

    messages = helper.create_message(
        data["image_frames"].flatten(0, 1),
        camera_indices=data.get("camera_indices"),
        num_frames_per_camera=int(data["image_frames"].shape[1]),
        nav_text=nav_text or None,
    )

    processor_kwargs = {}
    if vlm_image_pixels is not None and int(vlm_image_pixels) > 0:
        processor_kwargs = {
            "min_pixels": int(vlm_image_pixels),
            "max_pixels": int(vlm_image_pixels),
        }

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
        **processor_kwargs,
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    diffusion_kwargs = {"inference_step": 10}
    inference_fn = model.sample_trajectories_from_data_with_vlm_rollout
    use_cfg_nav = False
    if nav_text and not math.isclose(float(navigation_weight), 1.0):
        use_cfg_nav = True
        inference_fn = model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav
        diffusion_kwargs = {
            **diffusion_kwargs,
            "use_classifier_free_guidance": True,
            "inference_guidance_weight": float(navigation_weight),
        }

    disable_generate_logits = disable_unused_generate_logits and not use_cfg_nav
    _prime_oom_pipeline(model)
    with (
        optimized_vlm_generate(
            model,
            disable_output_logits=disable_generate_logits,
            timing=vlm_generate_timing,
        ),
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        pred_xyz, pred_rot, extra = inference_fn(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=cfg.NUM_TRAJ_SAMPLES,
            diffusion_kwargs=diffusion_kwargs,
            max_generation_length=256,
            return_extra=True,
        )

    return pred_xyz, extra


def _extract_text_field(extra, key):
    if not isinstance(extra, dict):
        return ""
    if key not in extra:
        return ""

    value = extra[key]
    if value is None:
        return ""

    while True:
        if isinstance(value, str):
            return str(value).strip()
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return ""
            value = value[0]
            continue
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return ""
            value = value.flat[0]
            continue
        if hasattr(value, "numel") and hasattr(value, "reshape"):
            if int(value.numel()) == 0:
                return ""
            value = value.reshape(-1)[0].item()
            continue
        return str(value).strip()


def _clean_generated_answer_text(text):
    text = SPECIAL_TOKEN_RE.sub("", str(text))
    return " ".join(text.split()).strip()


def _extract_answer_from_decoded_text(decoded_text):
    text = str(decoded_text).strip()
    if not text:
        return ""

    if "<|answer_end|>" in text:
        candidate = text.partition("<|answer_end|>")[0]
        if "<|answer_start|>" in candidate:
            candidate = candidate.rsplit("<|answer_start|>", 1)[1]
        return _clean_generated_answer_text(candidate)

    if "<|answer_start|>" in text:
        candidate = text.rsplit("<|answer_start|>", 1)[1]
    else:
        candidate = text

    for terminator in VQA_ANSWER_TERMINATORS:
        if terminator == "<|answer_end|>":
            continue
        candidate = candidate.split(terminator, 1)[0]

    return _clean_generated_answer_text(candidate)


def _generate_vqa_text_with_partial_answer_fallback(
    model,
    model_inputs,
    top_p=0.98,
    top_k=None,
    temperature=0.6,
    num_samples=1,
    max_generation_length=256,
):
    """Generate VQA text while preserving partial answers without ``answer_end``."""

    tokenized_data = dict(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_samples
    generation_config.max_new_tokens = max_generation_length
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = top_k
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    generated = model.vlm.generate(
        input_ids=input_ids,
        **tokenized_data,
        generation_config=generation_config,
    )
    sequences = generated["sequences"] if isinstance(generated, dict) else generated.sequences
    generated_tokens = sequences[:, input_ids.shape[1] :]
    decoded_batch = model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    batch_size = int(input_ids.shape[0])
    raw = np.array(decoded_batch, dtype=object).reshape([batch_size, num_samples])
    answers = np.array(
        [_extract_answer_from_decoded_text(text) for text in decoded_batch],
        dtype=object,
    ).reshape([batch_size, num_samples])
    return {
        "answer": answers,
        "raw_answer": raw,
    }


def extract_cot_text(extra):
    return _extract_text_field(extra, "cot")


def extract_answer_text(extra):
    answer = _extract_text_field(extra, "answer")
    if answer:
        return _extract_answer_from_decoded_text(answer)

    for key in ("raw_answer", "raw_text", "decoded_answer", "decoded_text"):
        raw_answer = _extract_text_field(extra, key)
        if raw_answer:
            return _extract_answer_from_decoded_text(raw_answer)

    return ""


def run_vqa(
    model,
    processor,
    data,
    question: str,
):
    """Run Alpamayo VQA text generation for a driving-relevant question."""

    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = helper.create_vqa_message(
        data["image_frames"].flatten(0, 1),
        question=question,
        camera_indices=data.get("camera_indices"),
        num_frames_per_camera=int(data["image_frames"].shape[1]),
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device({"tokenized_data": inputs}, "cuda")

    _prime_oom_pipeline(model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if hasattr(model, "vlm") and hasattr(model, "tokenizer"):
            return _generate_vqa_text_with_partial_answer_fallback(
                model,
                model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_samples=1,
                max_generation_length=256,
            )
        return model.generate_text(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_samples=1,
            max_generation_length=256,
        )


def extract_trajectory_samples(pred_xyz):
    arr = pred_xyz.detach().cpu().numpy()
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected pred_xyz shape after squeeze: {arr.shape}")
    if arr.shape[-1] < 3:
        raise ValueError(f"Trajectory last dim must be >= 3, got {arr.shape}")
    return arr[:, :, :3]


def select_trajectory_by_prev_similarity(traj_samples, prev_traj):
    num_samples = traj_samples.shape[0]
    if prev_traj is None:
        return 0, [None] * num_samples

    prev_xy = prev_traj[:, :2]
    scores = []
    for i in range(num_samples):
        curr_xy = traj_samples[i, :, :2]
        n = min(len(curr_xy), len(prev_xy))
        if n <= 0:
            score = float("inf")
        else:
            score = float(np.mean(np.linalg.norm(curr_xy[:n] - prev_xy[:n], axis=1)))
        scores.append(score)

    best_idx = int(np.argmin(scores))
    return best_idx, scores
