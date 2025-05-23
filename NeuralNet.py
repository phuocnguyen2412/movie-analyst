import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, dropout_rate):
        super(Net, self).__init__()
        layers = []
        hidden_dim = 512  # Kích thước ban đầu của lớp ẩn

        # Thêm các lớp ẩn động dựa trên `num_hidden_layers`
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim  # Đầu ra của lớp trước là đầu vào của lớp tiếp theo
            hidden_dim //= 2  # Giảm kích thước lớp ẩn (512 -> 256 -> 128 -> ...)

        # Lớp đầu ra
        layers.append(nn.Linear(input_dim, 1))

        # Tạo mạng từ danh sách các lớp
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)