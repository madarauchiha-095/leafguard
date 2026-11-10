"""Grad-CAM++ for CNN feature maps and DeiT patch-token activations."""
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.inference.common import CLASS_NAMES, make_transform


def target_layers(model, model_key="hybrid"):
    """Return architecture-aware targets; transformer activations are patch tokens."""
    if model_key == "hybrid":
        return {
            "mobilenet": model.mobilenet_features[-1],
            "deit": model.deit.blocks[-1].norm1,
        }
    if model_key in ("mobilenet", "mobilenet_clahe"):
        return {"mobilenet": model.model.features[-1]}
    if model_key == "deit_tiny":
        return {"deit": model.model.blocks[-1].norm1}
    raise ValueError(f"Unsupported model key: {model_key}")


def _spatial_activation(tensor):
    if tensor.ndim == 4:  # CNN: B,C,H,W
        return tensor
    if tensor.ndim != 3:  # ViT: B,tokens,channels
        raise ValueError(f"Expected CNN or token activation, got shape {tuple(tensor.shape)}")
    token_count = tensor.shape[1]
    # DeiT-Tiny uses one CLS token followed by 14x14 patch tokens.
    patch_count = token_count - 1
    side = int(patch_count ** 0.5)
    if side * side != patch_count:
        raise ValueError(f"Cannot form a square patch grid from {token_count} tokens")
    patches = tensor[:, 1:, :]
    return patches.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], side, side)


def _cam_from_activation(activation, gradient, output_size):
    activation = _spatial_activation(activation)
    gradient = _spatial_activation(gradient)
    grad2 = gradient.square()
    grad3 = grad2 * gradient
    denominator = 2.0 * grad2 + (activation * grad3).sum(dim=(2, 3), keepdim=True)
    alpha = grad2 / (denominator + torch.finfo(grad2.dtype).eps)
    weights = (alpha * F.relu(gradient)).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
    minimum = cam.amin(dim=(2, 3), keepdim=True)
    maximum = cam.amax(dim=(2, 3), keepdim=True)
    cam = (cam - minimum) / (maximum - minimum).clamp_min(1e-8)
    return cam[0, 0].detach().cpu().numpy()


def gradcampp(model, image_tensor, class_index=None, model_key="hybrid"):
    """Return the model score, probabilities and per-branch normalized CAMs."""
    layers = target_layers(model, model_key)
    activations = {}
    handles = []
    for name, layer in layers.items():
        handles.append(layer.register_forward_hook(
            lambda module, inputs, output, key=name: activations.__setitem__(key, output)
        ))
    try:
        model.eval()
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = image_tensor.device
        x = image_tensor.unsqueeze(0).to(model_device)
        logits = model(x)
        probs = logits.softmax(dim=1)[0]
        predicted = int(probs.argmax().item())
        target = predicted if class_index is None else int(class_index)
        scores = torch.autograd.grad(
            logits[0, target], list(activations.values()), allow_unused=False,
        )
        branch_cams = {}
        for (name, _), activation, gradient in zip(layers.items(), activations.values(), scores):
            branch_cams[name] = _cam_from_activation(
                activation, gradient, image_tensor.shape[-2:]
            )
        combined = np.mean(np.stack(list(branch_cams.values()), axis=0), axis=0)
        combined = (combined - combined.min()) / max(float(combined.max() - combined.min()), 1e-8)
        return {
            "class_index": target,
            "predicted_index": predicted,
            "confidence": float(probs[predicted].detach().item()),
            "probabilities": probs.detach().cpu().numpy(),
            "branch_cams": branch_cams,
            "combined_cam": combined.astype(np.float32),
        }
    finally:
        for handle in handles:
            handle.remove()


def tensor_to_rgb(image_tensor):
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    arr = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip((arr * std + mean) * 255.0, 0, 255).astype(np.uint8)


def overlay_cam(rgb, cam, alpha=0.42):
    import cv2
    heat = cv2.applyColorMap(np.uint8(np.clip(cam, 0, 1) * 255), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1.0 - alpha) * rgb + alpha * heat)


def explain_pil(model, image: Image.Image, model_key="hybrid", clahe=False):
    # Resize only once inside the preprocessing pipeline so the explanation
    # score matches ordinary inference exactly.
    rgb = image.convert("RGB")
    input_tensor = make_transform(clahe=clahe)(rgb)
    result = gradcampp(model, input_tensor, model_key=model_key)
    original = tensor_to_rgb(input_tensor)
    result["original"] = original
    result["overlay"] = overlay_cam(original, result["combined_cam"])
    result["class_name"] = CLASS_NAMES[result["predicted_index"]]
    result["top3"] = [
        {"class": CLASS_NAMES[int(i)], "confidence": float(result["probabilities"][int(i)])}
        for i in np.argsort(result["probabilities"])[::-1][:3]
    ]
    return result
