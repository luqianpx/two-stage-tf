
# This is the Grad-CAM class
# author:px
# date:2022-01-07
# version:1.0

import torch.nn.functional as F
import torch

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_gradient(grad):
            self.gradients = grad
        
        def forward_hook(module, input, output):
            self.activations = output
            output.register_hook(save_gradient)
        
        target_layer.register_forward_hook(forward_hook)

    def generate(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0), size=input_image.shape[2:], mode='trilinear', align_corners=False)
        cam = cam.squeeze(0) / cam.max()
        return cam

# Apply Grad-CAM on the last conv layer (layer4)
gradcam = GradCAM3D(classifier_model, classifier_model[0].layer4)
sample_data, _ = next(iter(test_loader))
sample_data = sample_data[0:1].to(device)  # Take one sample
heatmap = gradcam.generate(sample_data)
print("Grad-CAM heatmap generated:", heatmap.shape)