import torch
import torch.nn.functional as F

# Tạo feature map kích thước (N=1, C=1, H=2, W=2)
feature_map = torch.tensor([[[[1.0, 2.0],
                              [3.0, 5.0]]]])  # shape (1, 1, 2, 2)

# Tạo grid với 1 điểm: (-1, -1) chính là pixel (0, 0)
# shape của grid: (N=1, H_out=2, W_out=1, 2)
grid = torch.tensor([[
        [[-1.0, -1.0]], 
        [[0.0, 0.0]]
    ]])  # lấy điểm góc trên-trái

# Dùng grid_sample để sample
output = F.grid_sample(feature_map, grid, mode='bilinear', align_corners=True)

print("Feature map:\n", feature_map, feature_map.shape)
print("Sampled value at:\n", output, output.shape)