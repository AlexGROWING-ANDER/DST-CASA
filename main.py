import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dstt import DSTT
from alignment import SoftAlignment


def train(backbone, align_heads, train_loader, target_loader, optimizer, epoch, configs):
    backbone.train()
    align_heads.train()

    crit_cls = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    crit_recon = torch.nn.MSELoss()

    for (source_data, target_data) in zip(train_loader, target_loader):
        source_data, target_data = source_data.to(device), target_data.to(device)
        optimizer.zero_grad()

        h_s = backbone(source_data.x, source_data.batch, apply_mask=False)
        cls_s, pred_s, z_s, _ = align_heads(h_s)

        h_s_aug = backbone(source_data.x, source_data.batch, apply_mask=True)
        _, _, z_s_aug, _ = align_heads(h_s_aug)

        h_t = backbone(target_data.x, target_data.batch, apply_mask=False)
        cls_t, pred_t, z_t, _ = align_heads(h_t)

        h_t_aug = backbone(target_data.x, target_data.batch, apply_mask=True)
        _, _, z_t_aug, _ = align_heads(h_t_aug)

        loss_cls = crit_cls(cls_s, torch.argmax(source_data.y, dim=1))
        loss_mmd = calculate_mmd(h_s, h_t)
        loss_recon = calculate_recon_loss(h_s, h_t, z_s, z_t, source_data.x, target_data.x, align_heads.decoder)

        if epoch < 5:
            loss_soft = torch.tensor(0.0).to(device)
        else:
            loss_soft = calculate_soft_alignment_loss(
                z_s, z_s_aug, z_t, z_t_aug,
                source_data.y, pred_t.detach(),
                temperature=configs['temp']
            )

        total_loss = loss_cls + configs['alpha'] * loss_mmd + \
                     configs['beta'] * loss_soft + configs['gamma'] * loss_recon

        total_loss.backward()
        optimizer.step()


def main():
    print('=== DSTA: Training with DSTT and Soft Alignment ===')

    backbone = DSTT(num_channels=62, seq_len=1325).to(device)
    align_heads = SoftAlignment(d_model=256, original_input_dim=62 * 1325).to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(align_heads.parameters()),
        lr=5e-4, weight_decay=1e-4
    )

    for epoch in range(1000):
        train(backbone, align_heads, train_loader, target_loader, optimizer, epoch, configs)
        acc = evaluate(backbone, align_heads, test_loader)
        if acc > best_acc:
            best_acc = acc


if __name__ == '__main__':
    main()