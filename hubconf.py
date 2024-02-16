def custom(path="/fasterrcnn_model_drinks_Epoch9.pt", autoshape=True, _verbose=True, device=None):
    import torch
    import torchvision
    from pathlib import Path
    import os
    print(os.getcwd())
    print(os.listdir())
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    num_classes = 4             # 3 drinks + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    model_path = Path(path)      # edit epoch as needed

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)