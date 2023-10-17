import torch

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = torch.nn.Linear(20, 30).to(DEVICE)
    input = torch.randn(128, 20).to(DEVICE)
    output = m(input)
    print('output', output.size())
    exit()
