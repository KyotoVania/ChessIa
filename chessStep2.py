import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessAI(nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        # Définir les couches ici
        self.fc1 = nn.Linear(64, 128)  # 64 entrées pour chaque case, 128 neurones dans la couche cachée
        self.fc2 = nn.Linear(128, 64)  # 64 sorties pour chaque case

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

chess_model = ChessAI()


# Test du modèle
input_data = torch.rand(1, 64)  # Données d'entrée aléatoires pour un plateau d'échecs
output_data = chess_model(input_data)
print("Output:", output_data)
