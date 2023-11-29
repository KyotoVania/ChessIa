    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

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



    criterion = nn.MSELoss()
    optimizer = optim.SGD(chess_model.parameters(), lr=0.01)

    for epoch in range(100):  # nombre d'itérations
        input_data = torch.rand(10, 64)  # Données d'entrée aléatoires
        target_data = torch.rand(10, 64)  # Données cibles aléatoires

        optimizer.zero_grad()
        outputs = chess_model(input_data)
        loss = criterion(outputs, target_data)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
