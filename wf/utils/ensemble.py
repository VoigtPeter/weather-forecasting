import torch


def ensemble_batch(x: torch.Tensor, ensemble_size: int) -> torch.Tensor:
    # x: (batch, ...) -> (batch * ensemble, ...)
    return x.unsqueeze(1).expand(x.shape[0], ensemble_size, *x.shape[1:]).reshape(-1, *x.shape[1:])


def reverse_ensemble_batch(x: torch.Tensor, ensemble_size: int) -> torch.Tensor:
    # x: (batch * ensemble, ...) -> (batch, ensemble, ...)
    return x.view(-1, ensemble_size, *x.shape[1:])


if __name__ == "__main__":
    #x = torch.arange(3).view(-1, 1).expand(3, 5)
    x = torch.arange(5)
    print(x)
    print()
    ens = ensemble_batch(x, 2)
    print(ens)
    print()
    print(reverse_ensemble_batch(ens, 2))
