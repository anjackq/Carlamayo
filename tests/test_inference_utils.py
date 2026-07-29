import numpy as np
import pytest
import torch

from module import inference


def test_model_name_uses_alpamayo_15_weights():
    assert inference.ALPAMAYO_MODEL_NAME == "nvidia/Alpamayo-1.5-10B"


def test_require_cuda_runtime_fails_before_model_loading_without_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="Exclusive_Process") as exc_info:
        inference.require_cuda_runtime()

    assert "CUDA_VISIBLE_DEVICES=0" in str(exc_info.value)


def test_prepare_model_input_builds_expected_tensor_shapes_and_types():
    images = np.zeros((4, 4, 8, 12, 3), dtype=np.uint8)
    history_xyz = np.zeros((16, 3), dtype=np.float32)
    history_rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 16, axis=0)

    model_input = inference.prepare_model_input(images, history_xyz, history_rot)

    assert model_input["image_frames"].shape == (4, 4, 3, 8, 12)
    assert model_input["image_frames"].dtype == torch.uint8
    assert model_input["camera_indices"].dtype == torch.int64
    assert model_input["camera_indices"].device.type == "cpu"
    assert model_input["camera_indices"].tolist() == [0, 1, 2, 6]
    assert model_input["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert model_input["ego_history_xyz"].dtype == torch.float32
    assert model_input["ego_history_rot"].shape == (1, 1, 16, 3, 3)

    messages = inference.helper.create_message(
        model_input["image_frames"].flatten(0, 1),
        camera_indices=model_input["camera_indices"],
        num_frames_per_camera=4,
    )
    labels = [
        item["text"]
        for item in messages[1]["content"]
        if item["type"] == "text"
    ]
    assert "Front left camera: " in labels
    assert "Front camera: " in labels
    assert "Front right camera: " in labels
    assert "Front telephoto camera: " in labels
    assert labels.count("frame 0 ") == 4
    assert labels.count("frame 3 ") == 4


@pytest.mark.parametrize(
    ("images_shape", "history_shape", "rotation_shape", "message"),
    [
        ((3, 4, 8, 12, 3), (16, 3), (16, 3, 3), "images_array"),
        ((4, 3, 8, 12, 3), (16, 3), (16, 3, 3), "images_array"),
        ((4, 4, 8, 12, 3), (15, 3), (16, 3, 3), "history_xyz"),
        ((4, 4, 8, 12, 3), (16, 3), (15, 3, 3), "history_rot"),
    ],
)
def test_prepare_model_input_rejects_wrong_synchronized_shapes(
    images_shape,
    history_shape,
    rotation_shape,
    message,
):
    with pytest.raises(ValueError, match=message):
        inference.prepare_model_input(
            np.zeros(images_shape, dtype=np.uint8),
            np.zeros(history_shape, dtype=np.float32),
            np.zeros(rotation_shape, dtype=np.float32),
        )


def test_prepare_model_input_validates_explicit_camera_identity_order():
    images = np.zeros((4, 4, 8, 12, 3), dtype=np.uint8)
    history_xyz = np.zeros((16, 3), dtype=np.float32)
    history_rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 16, axis=0)

    model_input = inference.prepare_model_input(
        images,
        history_xyz,
        history_rot,
        camera_indices=(0, 1, 2, 6),
    )
    assert model_input["camera_indices"].tolist() == [0, 1, 2, 6]

    with pytest.raises(ValueError, match="camera_indices"):
        inference.prepare_model_input(
            images,
            history_xyz,
            history_rot,
            camera_indices=(1, 0, 2, 6),
        )


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"cot": [[[["  reason about lanes  "]]]]}, "  reason about lanes  "),
        ({"cot": np.array(["stop for light"], dtype=object)}, "stop for light"),
        ({"cot": torch.tensor([7])}, "7"),
        ({}, ""),
    ],
)
def test_extract_cot_text_handles_nested_common_return_shapes(extra, expected):
    assert inference.extract_cot_text(extra) == expected


def test_extract_cot_text_selects_matching_candidate_without_normalizing_text():
    extra = {"cot": np.array([[" first ", " second\n"]], dtype=object)}

    assert inference.extract_cot_text(extra, candidate_index=1) == " second\n"


def test_extract_cot_texts_preserves_candidate_alignment():
    extra = {"cot": np.array([[[" first ", " second\n", " third "]]], dtype=object)}

    assert inference.extract_cot_texts(extra, candidate_count=3) == [
        " first ",
        " second\n",
        " third ",
    ]


def test_extract_answer_text_removes_special_tokens_and_terminators():
    extra = {
        "answer": np.array(
            ["<|answer_start|>The traffic light is red.<|answer_end|><|im_end|>"],
            dtype=object,
        )
    }

    assert inference.extract_answer_text(extra) == "The traffic light is red."


def test_extract_trajectory_samples_squeezes_batch_axes_and_keeps_xyz_only():
    pred_xyz = torch.arange(1 * 1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 1, 2, 3, 4)

    samples = inference.extract_trajectory_samples(pred_xyz)

    assert samples.shape == (2, 3, 3)
    np.testing.assert_allclose(samples, pred_xyz.numpy()[0, 0, :, :, :3])


def test_select_trajectory_by_prev_similarity_prefers_closest_xy_path():
    previous = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    candidates = np.array(
        [
            [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
            [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]],
        ],
        dtype=np.float32,
    )

    best_idx, scores = inference.select_trajectory_by_prev_similarity(candidates, previous)

    assert best_idx == 1
    assert scores[1] < scores[0]
