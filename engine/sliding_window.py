import warnings
import torch
import torch.nn.functional as F



def resize(
    input: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    warning: bool = True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def slide_inference(
    nnet: torch.nn.Module,
    model_mode: str,
    iter: int,
    img: torch.Tensor,
    pos_mask: torch.Tensor,
    crop_size: tuple[int, int],
    stride: tuple[int, int],
    num_classes: int,
):
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = img.size()
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_pos_mask = pos_mask[:, :, y1:y2, x1:x2]
            # crop_seg_logit = nnet(crop_img)
            crop_seg_logit = nnet.test(model_mode, iter, [crop_img, crop_pos_mask])
            preds += F.pad(
                crop_seg_logit,
                (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
            )

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    if torch.onnx.is_in_onnx_export():
        # cast count_mat to constant while exporting to ONNX
        count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(
            device=img.device
        )
    preds = preds / count_mat
    return F.softmax(preds, dim=1)


def slide_inference_rescale(
    model: torch.nn.Module,
    img: torch.Tensor,
    crop_size: tuple[int, int],
    stride: tuple[int, int],
    num_classes: int,
    align_corners: bool,
    original_shape: tuple[int, int],
):
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = img.size()
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_seg_logit = model(crop_img)
            preds += F.pad(
                crop_seg_logit,
                (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
            )

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    if torch.onnx.is_in_onnx_export():
        # cast count_mat to constant while exporting to ONNX
        count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(
            device=img.device
        )
    preds = preds / count_mat
    preds = resize(
        preds,
        size=original_shape,
        mode="bilinear",
        align_corners=align_corners,
        warning=False,
    )
    return F.softmax(preds, dim=1)