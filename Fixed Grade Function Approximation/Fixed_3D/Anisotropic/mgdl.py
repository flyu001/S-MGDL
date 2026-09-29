import torch.nn as nn


class SixAffineBlock(nn.Module):

    def __init__(self, in_dim, width):
        super().__init__()
        self.affine_1 = nn.Linear(in_dim, width)
        self.activation_1 = nn.ReLU()
        self.affine_2 = nn.Linear(width, width)
        self.activation_2 = nn.ReLU()
        self.affine_3 = nn.Linear(width, width)
        self.activation_3 = nn.ReLU()
        self.affine_4 = nn.Linear(width, width)
        self.activation_4 = nn.ReLU()
        self.affine_5 = nn.Linear(width, width)
        self.activation_5 = nn.ReLU()
        self.affine_6 = nn.Linear(width, width)
        self.activation_6 = nn.ReLU()

    def forward(self, z):
        z = self.activation_1(self.affine_1(z))
        z = self.activation_2(self.affine_2(z))
        z = self.activation_3(self.affine_3(z))
        z = self.activation_4(self.affine_4(z))
        z = self.activation_5(self.affine_5(z))
        z = self.activation_6(self.affine_6(z))
        return z


class FourAffineBlock(nn.Module):

    def __init__(self, in_dim, width):
        super().__init__()
        self.affine_1 = nn.Linear(in_dim, width)
        self.activation_1 = nn.ReLU()
        self.affine_2 = nn.Linear(width, width)
        self.activation_2 = nn.ReLU()
        self.affine_3 = nn.Linear(width, width)
        self.activation_3 = nn.ReLU()
        self.affine_4 = nn.Linear(width, width)
        self.activation_4 = nn.ReLU()

    def forward(self, z):
        z = self.activation_1(self.affine_1(z))
        z = self.activation_2(self.affine_2(z))
        z = self.activation_3(self.affine_3(z))
        z = self.activation_4(self.affine_4(z))
        return z


class TwoAffineBlock(nn.Module):

    def __init__(self, in_dim, width):
        super().__init__()
        self.affine_1 = nn.Linear(in_dim, width)
        self.activation_1 = nn.ReLU()
        self.affine_2 = nn.Linear(width, width)
        self.activation_2 = nn.ReLU()

    def forward(self, z):
        z = self.activation_1(self.affine_1(z))
        z = self.activation_2(self.affine_2(z))
        return z


class FixedDepthMGDL3D(nn.Module):

    def __init__(self, width, max_grades):
        super().__init__()
        self.width = int(width)
        self.max_grades = int(max_grades)

        if self.max_grades != 3:
            raise ValueError("The 6+4+2 experiment requires MAX_GRADES = 3.")

        feature_blocks = [SixAffineBlock(3, self.width)]
        feature_blocks.append(FourAffineBlock(self.width, self.width))
        feature_blocks.append(TwoAffineBlock(self.width, self.width))

        output_layers = []
        for grade in range(1, self.max_grades + 1):
            output_layers.append(nn.Linear(self.width, 1))

        self.feature_blocks = nn.ModuleList(feature_blocks)
        self.output_layers = nn.ModuleList(output_layers)

    def h(self, x, grade):
        h_value = x
        for k in range(int(grade)):
            h_value = self.feature_blocks[k](h_value)
        return h_value

    def u_raw(self, x, grade):
        h_grade = self.h(x, grade)
        return self.output_layers[int(grade) - 1](h_grade)

    def freeze_for_grade(self, grade):
        for parameter in self.parameters():
            parameter.requires_grad = False

        for parameter in self.feature_blocks[int(grade) - 1].parameters():
            parameter.requires_grad = True
        for parameter in self.output_layers[int(grade) - 1].parameters():
            parameter.requires_grad = True
