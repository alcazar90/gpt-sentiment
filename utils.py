"""
    File: utils.py

"""

import torch

@torch.no_grad()
def estimate_loss(model, loader, loss_fn, device, return_acc=False):
    """ 
        Computa la pérdida promedio de un modelo en un loader de datos.
        Es posible computar el accuracy si se pasa el parámetro return_acc=True.
    """
    model.eval()
    losses = []
    targets = []
    preds = []
    for idx, batch in enumerate(loader):
        xb = batch[0].to(device)
        yb = batch[1].to(device)
        y_pred = model(xb)
        if return_acc:
            preds.append(y_pred.argmax(dim=1))
            targets.append(yb)
        loss = loss_fn(y_pred, yb)
        losses.append(loss.item())
    model.train()
    if return_acc:
        return torch.tensor(losses).mean().item(), (torch.cat(preds) == torch.cat(targets)).float().mean().item()
    return torch.tensor(losses).mean().item()

def collect_preds(model, loader, device):
    """ Recolecta las predicciones y etiquetas dado un modelo y un dataloader """
    model.eval()
    all_preds = torch.tensor([])
    all_targets = torch.tensor([])
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            all_preds = torch.cat((all_preds, preds.cpu()), dim=0)
            all_targets = torch.cat((all_targets, yb.cpu()), dim=0)
    return all_preds, all_targets