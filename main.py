import torch
import torch.optim as optim
import torch.nn as nn
from model import CNN
from train import train
from test import test
from dataset import load_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar os dados
    train_loader, test_loader = load_data()

    # Configurar o modelo, otimizador e função de perda
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Treinamento e teste
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()
