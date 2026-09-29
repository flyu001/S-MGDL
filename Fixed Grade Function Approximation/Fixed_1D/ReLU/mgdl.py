import torch.nn as nn

class TwoAffineBlock(nn.Module):

    def __init__(self, in_dim, width):
        super().__init__()
        self.affine_1 = nn.Linear(in_dim, width)
        self.activation_1 = nn.ReLU()
        self.affine_2 = nn.Linear(width, width)
        self.activation_2 = nn.ReLU()

    def forward(self, z):
        z = self.affine_1(z)
        z = self.activation_1(z)
        z = self.affine_2(z)
        z = self.activation_2(z)
        return z

class FixedDepthMGDL1D(nn.Module):

    def __init__(self, width, max_grades):
        super().__init__()
        self.width = int(width)
        self.max_grades = int(max_grades)

        feature_blocks = []
        for grade in range(1, self.max_grades + 1):
            if grade == 1:
                in_dim = 1
            else:
                in_dim = self.width
            feature_blocks.append(TwoAffineBlock(in_dim, self.width))

        output_layers = []
        for grade in range(1, self.max_grades + 1):
            output_layers.append(nn.Linear(self.width, 1))

        self.feature_blocks = nn.ModuleList(feature_blocks)
        self.output_layers = nn.ModuleList(output_layers)

    def h(self, x, grade):
        h_value = x
        for k in range(grade):
            h_value = self.feature_blocks[k](h_value)
        return h_value

    def u_raw(self, x, grade):
        h_grade = self.h(x, grade)
        return self.output_layers[grade - 1](h_grade)

    def freeze_for_grade(self, grade):
        for parameter in self.parameters():
            parameter.requires_grad = False

        for parameter in self.feature_blocks[grade - 1].parameters():
            parameter.requires_grad = True
        for parameter in self.output_layers[grade - 1].parameters():
            parameter.requires_grad = True
