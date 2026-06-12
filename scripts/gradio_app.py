# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


MASK_COLORS = np.array(
    [
        [0, 114, 178],
        [213, 94, 0],
        [0, 158, 115],
        [204, 121, 167],
        [230, 159, 0],
        [86, 180, 233],
    ],
    dtype=np.float32,
)


def get_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def as_rgb_array(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise gr.Error("Upload an image first.")

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.shape[2] == 4:
        alpha = array[:, :, 3:4].astype(np.float32) / 255.0
        array = array[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    elif array.shape[2] > 3:
        array = array[:, :, :3]

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def parse_box(box_text: str) -> Optional[np.ndarray]:
    box_text = (box_text or "").strip()
    if not box_text:
        return None
    parts = [part.strip() for part in box_text.replace(" ", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise gr.Error("Box must contain four numbers: x1,y1,x2,y2.")
    try:
        x1, y1, x2, y2 = [float(part) for part in parts]
    except ValueError as exc:
        raise gr.Error("Box values must be numbers.") from exc
    if x2 <= x1 or y2 <= y1:
        raise gr.Error("Box must satisfy x2 > x1 and y2 > y1.")
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_from_corners(first: Sequence[int], second: Sequence[int]) -> np.ndarray:
    x1, x2 = sorted([float(first[0]), float(second[0])])
    y1, y2 = sorted([float(first[1]), float(second[1])])
    if x1 == x2 or y1 == y2:
        raise gr.Error("Box needs two different corners.")
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def format_box(box: np.ndarray) -> str:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    return f"{x1},{y1},{x2},{y2}"


def draw_prompts(
    image: np.ndarray,
    point_coords: Sequence[Sequence[int]],
    point_labels: Sequence[int],
    box: Optional[np.ndarray] = None,
) -> np.ndarray:
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    radius = max(4, round(max(image.shape[:2]) / 160))
    if box is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 210, 0), width=max(2, radius // 2))
    for (x, y), label in zip(point_coords, point_labels):
        color = (0, 190, 90) if label == 1 else (220, 45, 45)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )
    return np.asarray(canvas)


def overlay_prompt_mask(
    image: np.ndarray,
    mask: np.ndarray,
    point_coords: Sequence[Sequence[int]],
    point_labels: Sequence[int],
    box: Optional[np.ndarray],
    alpha: float,
) -> np.ndarray:
    color = np.array([0, 114, 178], dtype=np.float32)
    output = image.astype(np.float32).copy()
    output[mask] = output[mask] * (1.0 - alpha) + color * alpha
    return draw_prompts(np.clip(output, 0, 255).astype(np.uint8), point_coords, point_labels, box)


def overlay_auto_masks(image: np.ndarray, masks: List[dict], alpha: float) -> np.ndarray:
    output = image.astype(np.float32).copy()
    for index, ann in enumerate(sorted(masks, key=lambda item: item["area"], reverse=True)):
        color = MASK_COLORS[index % len(MASK_COLORS)]
        mask = ann["segmentation"]
        output[mask] = output[mask] * (1.0 - alpha) + color * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


class SamGradioRunner:
    def __init__(self, checkpoint: str, model_type: str, device: str) -> None:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.device = get_device(device)
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def predict_from_prompts(
        self,
        image: np.ndarray,
        point_coords: Sequence[Sequence[int]],
        point_labels: Sequence[int],
        box_text: str,
        multimask_output: bool,
        mask_index: int,
        alpha: float,
    ) -> Tuple[np.ndarray, str]:
        rgb = as_rgb_array(image)
        box = parse_box(box_text)

        if not point_coords and box is None:
            return draw_prompts(rgb, point_coords, point_labels, box), "Add a point or box prompt."

        coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        labels = np.array(point_labels, dtype=np.int32) if point_labels else None

        self.predictor.set_image(rgb)
        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=box,
            multimask_output=multimask_output,
        )

        selected_index = int(np.clip(mask_index, 0, len(masks) - 1))
        if not multimask_output:
            selected_index = 0
        overlay = overlay_prompt_mask(
            rgb,
            masks[selected_index].astype(bool),
            point_coords,
            point_labels,
            box,
            alpha,
        )
        status = f"Mask {selected_index}; predicted IoU: {scores[selected_index]:.3f}; device: {self.device}"
        return overlay, status

    def generate_automatic_masks(self, image: np.ndarray, alpha: float) -> Tuple[np.ndarray, str]:
        rgb = as_rgb_array(image)
        masks = self.mask_generator.generate(rgb)
        if not masks:
            return rgb, f"No masks found; device: {self.device}"
        overlay = overlay_auto_masks(rgb, masks, alpha)
        return overlay, f"Generated {len(masks)} masks; device: {self.device}"


def build_app(runner: SamGradioRunner) -> gr.Blocks:
    with gr.Blocks(title="SAM Gradio Demo") as demo:
        gr.Markdown("# Segment Anything")

        point_coords_state = gr.State([])
        point_labels_state = gr.State([])
        box_corner_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                input_image = gr.Image(label="Image", type="numpy", interactive=True)
                output_image = gr.Image(label="SAM output", type="numpy", interactive=False)
            with gr.Column(scale=2):
                point_mode = gr.Radio(
                    choices=["Foreground", "Background", "Box"],
                    value="Foreground",
                    label="Click mode",
                )
                box_text = gr.Textbox(
                    label="Box prompt",
                    placeholder="Optional: x1,y1,x2,y2",
                )
                multimask_output = gr.Checkbox(value=True, label="Return multiple masks")
                mask_index = gr.Slider(0, 2, value=0, step=1, label="Mask index")
                alpha = gr.Slider(0.1, 0.9, value=0.55, step=0.05, label="Overlay opacity")
                with gr.Row():
                    run_prompt_button = gr.Button("Run prompt", variant="primary")
                    auto_button = gr.Button("Auto masks")
                clear_button = gr.Button("Clear prompts")
                status = gr.Textbox(label="Status", interactive=False)

        def add_point(
            image: np.ndarray,
            mode: str,
            point_coords: List[List[int]],
            point_labels: List[int],
            box_corner: Optional[List[int]],
            current_box_text: str,
            current_multimask_output: bool,
            current_mask_index: int,
            current_alpha: float,
            evt: gr.SelectData,
        ) -> Tuple[List[List[int]], List[int], Optional[List[int]], str, np.ndarray, str]:
            if image is None:
                raise gr.Error("Upload an image first.")
            index = getattr(evt, "index", None)
            if not isinstance(index, (list, tuple)) or len(index) < 2:
                raise gr.Error("Could not read the clicked image coordinate.")

            x, y = int(index[0]), int(index[1])
            if mode == "Box":
                if box_corner is None:
                    rgb = as_rgb_array(image)
                    preview = draw_prompts(rgb, point_coords or [], point_labels or [], parse_box(current_box_text))
                    return (
                        point_coords or [],
                        point_labels or [],
                        [x, y],
                        current_box_text,
                        preview,
                        "Box start set. Click the opposite corner.",
                    )

                box = box_from_corners(box_corner, [x, y])
                next_box_text = format_box(box)
                overlay, next_status = runner.predict_from_prompts(
                    image,
                    point_coords or [],
                    point_labels or [],
                    next_box_text,
                    current_multimask_output,
                    current_mask_index,
                    current_alpha,
                )
                return (
                    point_coords or [],
                    point_labels or [],
                    None,
                    next_box_text,
                    overlay,
                    next_status,
                )

            next_coords = list(point_coords or []) + [[x, y]]
            next_labels = list(point_labels or []) + [1 if mode == "Foreground" else 0]
            overlay, next_status = runner.predict_from_prompts(
                image,
                next_coords,
                next_labels,
                current_box_text,
                current_multimask_output,
                current_mask_index,
                current_alpha,
            )
            return next_coords, next_labels, box_corner, current_box_text, overlay, next_status

        def run_prompt(
            image: np.ndarray,
            point_coords: List[List[int]],
            point_labels: List[int],
            current_box_text: str,
            current_multimask_output: bool,
            current_mask_index: int,
            current_alpha: float,
        ) -> Tuple[np.ndarray, str]:
            return runner.predict_from_prompts(
                image,
                point_coords or [],
                point_labels or [],
                current_box_text,
                current_multimask_output,
                current_mask_index,
                current_alpha,
            )

        def clear_prompts(
            image: np.ndarray,
        ) -> Tuple[List[List[int]], List[int], None, str, np.ndarray, str]:
            rgb = as_rgb_array(image)
            return [], [], None, "", rgb, "Cleared prompts."

        input_image.select(
            add_point,
            inputs=[
                input_image,
                point_mode,
                point_coords_state,
                point_labels_state,
                box_corner_state,
                box_text,
                multimask_output,
                mask_index,
                alpha,
            ],
            outputs=[
                point_coords_state,
                point_labels_state,
                box_corner_state,
                box_text,
                output_image,
                status,
            ],
        )
        run_prompt_button.click(
            run_prompt,
            inputs=[
                input_image,
                point_coords_state,
                point_labels_state,
                box_text,
                multimask_output,
                mask_index,
                alpha,
            ],
            outputs=[output_image, status],
        )
        auto_button.click(
            runner.generate_automatic_masks,
            inputs=[input_image, alpha],
            outputs=[output_image, status],
        )
        clear_button.click(
            clear_prompts,
            inputs=[input_image],
            outputs=[
                point_coords_state,
                point_labels_state,
                box_corner_state,
                box_text,
                output_image,
                status,
            ],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Segment Anything with a Gradio UI.")
    parser.add_argument("--checkpoint", required=True, help="Path to a SAM checkpoint .pth file.")
    parser.add_argument(
        "--model-type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h", "default"],
        help="SAM model type matching the checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps. Defaults to auto.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", default=7860, type=int, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = SamGradioRunner(args.checkpoint, args.model_type, args.device)
    demo = build_app(runner)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
