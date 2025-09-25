# main file to solve problem 1

from p1.utils.wrappers import YOLOWrapper, RFWrapper, EnsembleWrapper
from p1.utils.metric import MetricEvaluator
from p1.utils.helper import get_images_labels, preprocess_json


def main():
    image_paths, label_paths = get_images_labels(
        purpose="val",
        debug=True,
    )
    image_paths, label_paths = image_paths[:100], label_paths[:100]  # for quick testing

    model_RF = RFWrapper(method="nms", use_tta=True)
    model_YOLO = YOLOWrapper(method="nms", use_tta=True)
    ensemble_model = EnsembleWrapper([model_RF, model_YOLO], method="wbf", use_tta=True)
    metrichandler = MetricEvaluator()

    for image_path, label_path in zip(image_paths, label_paths):
        image_path = [image_path]
        label_path = [label_path]
        labels = preprocess_json(label_path)  # list of Tensors
        preds = ensemble_model.predict(image_path)  # list of (boxes, scores)
        processed_preds = metrichandler.process_preds(preds, type="xyxy")
        metrichandler.update(processed_preds, labels)
        result = metrichandler.compute()
        print("mAP result:", result["map_50"], result["map"])


def main2():
    model_RF = RFWrapper(method="nms", use_tta=True)
    model_YOLO = YOLOWrapper(method="nms", use_tta=True)
    ensemble_model = EnsembleWrapper([model_RF, model_YOLO], method="wbf", use_tta=True)
    sample_image_path_1 = "data/VS_KS/K3_CHN_20130308050237_30.jpg"
    sample_image_path_2 = "data/VS_KS/K3_CHN_20130308050353_17.jpg"
    sample_image_path_3 = "data/VS_KS/K3A_CHN_20240329055658_26.jpg"
    # results_RF = model_RF.predict(
    #     [sample_image_path_1, sample_image_path_2, sample_image_path_3]
    # )
    # results_YOLO = model_YOLO.predict(
    #     [sample_image_path_1, sample_image_path_2, sample_image_path_3]
    # )
    # print("Predictions RF:", results_RF)
    # print("Predictions YOLO:", results_YOLO)
    results_ensemble = ensemble_model.predict(
        [sample_image_path_1, sample_image_path_2, sample_image_path_3]
    )
    print("Predictions Ensemble:", results_ensemble)


if __name__ == "__main__":
    main()
