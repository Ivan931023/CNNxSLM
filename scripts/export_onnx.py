import torch
import torch.nn as nn
import os

class ZernikeResNet(nn.Module):
    def __init__(self, num_predict_modes=14):
        super().__init__()
        import torchvision.models as models
        self.resnet = models.resnet34(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_predict_modes)
        
    def forward(self, x): 
        return self.resnet(x)

def main():
    model_path = os.path.join('../outputs', 'models', 'best_resnet_finetuned.pth')
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}")
        return

    print("Loading PyTorch model...")
    device = torch.device('cpu')
    model = ZernikeResNet(num_predict_modes=14).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Creating dummy input [1, 1, 500, 500]...")
    dummy_input = torch.randn(1, 1, 500, 500, device=device)

    onnx_path = os.path.join('../outputs', 'models', 'zernike_resnet.onnx')
    print(f"Exporting to {onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ Successfully exported ONNX model!")

if __name__ == '__main__':
    main()
