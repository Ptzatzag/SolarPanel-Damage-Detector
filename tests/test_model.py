from functools import partial

from model import maskrcnn


def test_model_has_correct_number_of_classes(monkeypatch):
    # Test the architecture without downloading pretrained backbone weights.
    monkeypatch.setattr(
        maskrcnn,
        "maskrcnn_resnet50_fpn",
        partial(maskrcnn.maskrcnn_resnet50_fpn, weights_backbone=None),
    )
    model = maskrcnn.get_model(num_classes=3)

    assert model.roi_heads.box_predictor.cls_score.out_features == 3
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 3
