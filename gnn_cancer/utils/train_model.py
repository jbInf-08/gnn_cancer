import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb

def train_model(model, optimizer, scheduler, train_loader, val_loader, model_type, config, device):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        predictions = []
        labels = []
        logits = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                val_loss += F.cross_entropy(out, batch.y).item()
                pred = out.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
                labels.extend(batch.y.cpu().numpy())
                logits.extend(out.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_metric = calculate_metrics(predictions, labels, logits=logits)
        train_metric = calculate_metrics(predictions, labels)
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"results/{model_type}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_metric['accuracy'],
            'val_f1': val_metric['f1'],
            'val_precision': val_metric['precision'],
            'val_recall': val_metric['recall'],
            'val_loss_metric': val_metric['loss']
        })

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

def calculate_metrics(predictions, labels, logits=None):
    acc = np.mean(np.array(predictions) == np.array(labels))
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    loss = None
    if logits is not None:
        loss = F.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'loss': loss
    } 