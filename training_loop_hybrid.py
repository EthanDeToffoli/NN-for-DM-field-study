import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt

from modelCNN import (
    LogNormalDataset,
    IbridaCosmoResNet,
    SimpleIbridaCosmoNet,
    IbridaCosmoResNet50,
)


def aggiorna_grafico_loss(
    storico_train, storico_val, storico_train_acc, storico_val_acc, percorso_plot
):
    plt.figure(figsize=(10, 6))
    plt.plot(storico_train, label="Train Loss", color="blue", linewidth=2)
    plt.plot(storico_val, label="Validation Loss", color="red", linewidth=2)

    # Trova l'epoca migliore e i valori in quel punto
    miglior_loss_val = min(storico_val)
    epoca_migliore = storico_val.index(miglior_loss_val)

    miglior_loss_train = storico_train[epoca_migliore]
    miglior_acc_train = storico_train_acc[epoca_migliore]
    miglior_acc_val = storico_val_acc[epoca_migliore]

    # Disegna il punto del modello migliore
    plt.scatter(
        epoca_migliore,
        miglior_loss_val,
        color="green",
        s=100,
        zorder=5,
        label="Best Model",
    )

    # Crea il testo da inserire nella "bandierina"
    testo_bandiera = (
        f"Epoch: {epoca_migliore}\n"
        f"Train Loss: {miglior_loss_train:.4f} | R²: {miglior_acc_train:.2f}\n"
        f"Val Loss: {miglior_loss_val:.4f} | R²: {miglior_acc_val:.2f}"
    )

    # Aggiungi l'annotazione (bandierina)
    plt.annotate(
        testo_bandiera,
        xy=(epoca_migliore, miglior_loss_val),  # Punto a cui punta la freccia
        xytext=(15, 40),  # Posizione del testo (offset rispetto al punto)
        textcoords="offset points",  # Usa offset in pixel
        bbox=dict(
            boxstyle="round,pad=0.5", fc="lightyellow", ec="black", lw=1, alpha=0.9
        ),  # Stile del box
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black"
        ),  # Stile freccia
        fontsize=9,
        zorder=10,
    )

    plt.title("Learning Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Standardized)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(percorso_plot, bbox_inches="tight")
    plt.close()


def main():
    # PERCORSO_X = '/mnt/dataset_lognormale/campi_x_3param.pt'
    # PERCORSO_Y = '/mnt/dataset_lognormale/etichette_y_3param.pt'
    # PERCORSO_MODELLO = '/mnt/dataset_lognormale/modello_addestrato_3param.pth'

    # PERCORSO_X = '/mnt/dataset_lognormale/campi_x_1000&256.pt'
    # PERCORSO_Y = '/mnt/dataset_lognormale/etichette_y_1000&256.pt'
    # PERCORSO_MODELLO = '/mnt/dataset_lognormale/modello_addestrato_1000&256.pth'

    # PERCORSO_X = '/mnt/dataset_lognormale/campi_x_ns_only.pt'
    # PERCORSO_Y = '/mnt/dataset_lognormale/etichette_y_ns_only.pt'
    # PERCORSO_MODELLO = '/mnt/dataset_lognormale/modello_addestrato_ns_only.pth'

    # PERCORSO_X = '/mnt/dataset_lognormale/campi_x_big.pt'
    # PERCORSO_Y = '/mnt/dataset_lognormale/etichette_y_big.pt'
    # PERCORSO_MODELLO = '/mnt/dataset_lognormale/modello_addestrato_big.pth'

    # PERCORSO_X = '/mnt/dataset_lognormale/campi_x_base.pt'
    # PERCORSO_Y = '/mnt/dataset_lognormale/etichette_y_base.pt'
    # PERCORSO_Z = '/mnt/dataset_lognormale/pk_z_base.pt'
    # PERCORSO_MODELLO = '/mnt/dataset_lognormale/modello_addestrato_base.pth'

    PERCORSO_X = "/mnt/dataset_lognormale/big_data_campi.pt"
    PERCORSO_Y = "/mnt/dataset_lognormale/big_data_labels.pt"
    PERCORSO_Z = "/mnt/dataset_lognormale/big_data_pk.pt"
    PERCORSO_MODELLO = "/mnt/dataset_lognormale/modello_addestrato_big_data.pth"

    BATCH_SIZE = 128
    EPOCHE = 70
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inizio addestramento su hardware: {device}\n")

    print("Caricamento dataset già standardizzato")
    dataset_completo = LogNormalDataset(PERCORSO_X, PERCORSO_Y, PERCORSO_Z)

    dim_totale = len(dataset_completo)
    dim_train = int(0.7 * dim_totale)
    dim_val = int(0.15 * dim_totale)
    dim_test = dim_totale - dim_train - dim_val

    generatore = torch.Generator().manual_seed(
        42
    )  # seed manuale per avere sempre stessa divisione
    dati_train, dati_val, dati_test = random_split(
        dataset_completo, [dim_train, dim_val, dim_test], generator=generatore
    )

    train_loader = DataLoader(dati_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dati_val, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {dim_train} | Validation: {dim_val} | Test: {dim_test}\n")

    lunghezza_pk = dataset_completo.pk_data.shape[1]
    modello = IbridaCosmoResNet(pk_length=lunghezza_pk).to(device)
    ottimizzatore = optim.Adam(modello.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        ottimizzatore,
        mode="min",
        factor=0.3,
        patience=5,
    )

    funzione_perdita = nn.MSELoss()  # forse ci sono scelte migliori?

    if os.path.exists(PERCORSO_MODELLO):
        print(f"Trovato modello salvato in {PERCORSO_MODELLO}!")
        modello.load_state_dict(
            torch.load(PERCORSO_MODELLO, map_location=device, weights_only=True)
        )
        print("L'addestramento riprenderà da dove si era interrotto.\n")
    else:
        print("Nessun modello precedente trovato: partenza da zero.\n")

    miglior_loss_val = float(
        "inf"
    )  # serve per trovare il minimo della loss iterativamente

    storia_train_loss = []
    storia_val_loss = []
    storia_train_acc = []
    storia_val_acc = []

    PERCORSO_GRAFICO_LOSS = "/home/ubuntu/andamento_loss.png"

    for epoca in range(EPOCHE):
        modello.train()
        loss_train_totale = 0.0

        # variabili per calcolare l'accuratezza (R2 Score) globale dell'epoca
        ss_res_train = 0.0
        ss_tot_train = 0.0

        for batch_x, batch_pk, batch_y in train_loader:
            batch_x, batch_pk, batch_y = (
                batch_x.to(device),
                batch_pk.to(device),
                batch_y.to(device),
            )

            ottimizzatore.zero_grad()
            previsioni = modello(batch_x, batch_pk)

            loss = funzione_perdita(previsioni, batch_y)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                modello.parameters(), max_norm=1.0
            )  # max_norm è la massima norma L2 di tutti i gradienti sommati
            ottimizzatore.step()

            loss_train_totale += loss.item() * batch_x.size(0)

            with torch.no_grad():
                ss_res_train += torch.sum(
                    (batch_y - previsioni) ** 2
                ).item()  # discostamento previsioni da labels
                ss_tot_train += torch.sum(
                    (batch_y - torch.mean(batch_y, dim=0)) ** 2
                ).item()  # varianza dei labels

        media_loss_train = loss_train_totale / dim_train
        acc_train = max(
            0.0, 1 - (ss_res_train / (ss_tot_train + 1e-8))
        )  # taglia a 0 se negativo
        storia_train_loss.append(media_loss_train)
        storia_train_acc.append(acc_train)

        modello.eval()
        loss_val_totale = 0.0
        ss_res_val = 0.0
        ss_tot_val = 0.0

        with torch.no_grad():
            for batch_x, batch_pk, batch_y in val_loader:
                batch_x, batch_pk, batch_y = (
                    batch_x.to(device),
                    batch_pk.to(device),
                    batch_y.to(device),
                )
                previsioni = modello(batch_x, batch_pk)

                loss = funzione_perdita(previsioni, batch_y)
                loss_val_totale += loss.item() * batch_x.size(0)

                ss_res_val += torch.sum((batch_y - previsioni) ** 2).item()
                ss_tot_val += torch.sum(
                    (batch_y - torch.mean(batch_y, dim=0)) ** 2
                ).item()

        media_loss_val = loss_val_totale / dim_val
        acc_val = max(0.0, 1 - (ss_res_val / (ss_tot_val + 1e-8)))
        storia_val_loss.append(media_loss_val)
        storia_val_acc.append(acc_val)

        lr_corrente = ottimizzatore.param_groups[0]["lr"]

        aggiorna_grafico_loss(
            storia_train_loss,
            storia_val_loss,
            storia_train_acc,
            storia_val_acc,
            PERCORSO_GRAFICO_LOSS,
        )

        print(
            f"Epoca [{epoca + 1}/{EPOCHE}] | | LR: {lr_corrente:.4e} | "
            f"Train Loss: {media_loss_train:.4f} (Acc: {acc_train:.2f}) | "
            f"Val Loss: {media_loss_val:.4f} (Acc: {acc_val:.2f})"
        )

        if media_loss_val < miglior_loss_val:
            miglior_loss_val = media_loss_val
            torch.save(modello.state_dict(), PERCORSO_MODELLO)
            print(f"Modello e Grafico salvati.")

        scheduler.step(media_loss_val)


if __name__ == "__main__":
    main()
