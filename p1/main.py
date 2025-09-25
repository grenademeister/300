# main file to solve problem 1

from p1.utils.wrappers import YOLOWrapper, RFWrapper, EnsembleWrapper
from p1.utils.metric import MetricEvaluator
from p1.RT_DETR.wbf import RTWrapper
from p1.utils.helper import get_images_labels, preprocess_json


def main():
    image_paths, label_paths = get_images_labels(
        purpose="val",
        debug=True,
    )
    image_paths, label_paths = image_paths[:1], label_paths[:1]  # for quick testing

    model_YOLO = YOLOWrapper(method="nms", use_tta=True)
    model_RF = RFWrapper(method="nms", use_tta=True)
    model_RT = RTWrapper(
        method="nms",
        use_tta=True,
        checkpoint="fill_in_your_checkpoint_path",
    )
    ensemble_model = EnsembleWrapper(
        [model_RF, model_YOLO, model_RT], method="wbf", use_tta=True
    )
    metrichandler = MetricEvaluator(preprocess_preds=True, preprocess_targets=True)
    preds = ensemble_model.predict(image_paths)  # list of (boxes, scores)
    metrichandler.update(preds, label_paths)

    result = metrichandler.compute()
    print("mAP result:", result["map_50"], result["map"])


if __name__ == "__main__":
    main()
