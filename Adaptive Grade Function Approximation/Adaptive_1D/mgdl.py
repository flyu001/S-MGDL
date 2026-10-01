import torch.nn as nn

class ThreeAffineBlock(nn.Module):

    def __init__(self, in_dim, width):
        super().__init__()
        self.affine_1 = nn.Linear(in_dim, width)
        self.activation_1 = nn.ReLU()
        self.affine_2 = nn.Linear(width, width)
        self.activation_2 = nn.ReLU()
        self.affine_3 = nn.Linear(width, width)
        self.activation_3 = nn.ReLU()

    def forward(self, z):
        z = self.affine_1(z)
        z = self.activation_1(z)
        z = self.affine_2(z)
        z = self.activation_2(z)
        z = self.affine_3(z)
        z = self.activation_3(z)
        return z

class AutoDepthMGDL1D(nn.Module):

    def __init__(self, width, input_dim=1):
        super().__init__()
        self.width = int(width)
        self.input_dim = int(input_dim)
        self.feature_blocks = nn.ModuleList()
        self.output_layers = nn.ModuleList()

    def num_grades(self):
        return len(self.feature_blocks)

    def add_grade(self):
        if self.num_grades() == 0:
            in_dim = self.input_dim
        else:
            in_dim = self.width

        self.feature_blocks.append(ThreeAffineBlock(in_dim, self.width))
        self.output_layers.append(nn.Linear(self.width, 1))
        return self.num_grades()

    def remove_last_grade(self):
        if self.num_grades() == 0:
            raise ValueError("Cannot remove a grade from an empty MGDL model.")
        del self.feature_blocks[-1]
        del self.output_layers[-1]

    def h(self, x, grade):
        h_value = x
        for k in range(grade):
            h_value = self.feature_blocks[k](h_value)
        return h_value

    def u_raw(self, x, grade):
        h_grade = self.h(x, grade)
        return self.output_layers[grade - 1](h_grade)

    def Phi(self, x, grade):
        value = None
        h_value = x
        for k in range(grade):
            h_value = self.feature_blocks[k](h_value)
            u_k = self.output_layers[k](h_value)
            if value is None:
                value = u_k
            else:
                value = value + u_k
        return value

    def freeze_for_grade(self, grade):
        for parameter in self.parameters():
            parameter.requires_grad = False

        for parameter in self.feature_blocks[grade - 1].parameters():
            parameter.requires_grad = True
        for parameter in self.output_layers[grade - 1].parameters():
            parameter.requires_grad = True
