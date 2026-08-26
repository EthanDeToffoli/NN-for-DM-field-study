import numpy as np
import torch
import camb
import powerbox as pbox
from scipy.interpolate import interp1d
from scipy.stats import qmc
from tqdm import tqdm
import os



COSMOLGIE_NUM = 150       # Punti nel LH
UNIVERSI_NUM = 10000     
RESOLUTION = 256              
BOX_LENGTH = 1000.0          # Mpc

# limiti dei parametri [Omega_m, n_s, sigma_8]
LIMITI_INF = [0.10, 0.90, 0.60]
LIMITI_SUP = [0.50, 1.10, 1.00]

# percorsi di salvataggio
PERCORSO_X = '/mnt/dataset_lognormale/campi_x_base.pt'
PERCORSO_Y = '/mnt/dataset_lognormale/etichette_y_base.pt'
PERCORSO_Z = '/mnt/dataset_lognormale/pk_z_base.pt'


def genera_pk_cosmologico(omega_m, n_s, sigma8):

    params = camb.CAMBparams()
    
    H0_fixed = 67.5         # Km/(s*Mpc)
    omega_b_fixed = 0.049   # %             # valori da Planck
    
    h = H0_fixed / 100.0    
    ombh2 = omega_b_fixed * (h**2)  # densità fisica barioni -> omega_b*(h**2)
    omch2 = (omega_m - omega_b_fixed) * (h**2)  # densità fisica cdm -> omega_c*(h**2)
    
   
    params.set_cosmology(H0=H0_fixed, ombh2=ombh2, omch2=omch2)
    params.InitPower.set_params(ns=n_s) # A_s default, cambio dopo aver generato pk
    
    params.set_matter_power(redshifts=[0.0], kmax=10.0)
    params.NonLinear = camb.model.NonLinear_none    # non linearità data da trasf. log-normal (paper Coles e Jones)
    
    params.set_for_lmax(2500, lens_potential_accuracy=0)    # 2500 è quello di Planck, serve per inzializzare
                                                            # accuratezza 0 perché non mi intressa davvero calcolarlo
    
  

    results = camb.get_results(params)
    
    sigma8_default = results.get_sigma8()[0]        # vedo quale sigma_8 è stato generato casualmente
    sigma8_factor = (sigma8 / sigma8_default)**2    # fattore correzione a posteriori
    
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints=500)    # array dei k, dei redshift, del pk per ogni k (sarebbe matrice ma c'è solo un redshift)
                                                                                        # 500 punti sono 100/decade e funzionano bene almeno visivamente (test_pk.py) 
    pk_scaled = pk[0] * sigma8_factor

    k_mpc = kh * h

    pk_2d = pk_scaled / (h**3 * BOX_LENGTH)
    
    pk_interp = interp1d(k_mpc, pk_2d, kind='cubic', fill_value="extrapolate")    # fit con scipy.interpolate
    
    return pk_interp


def main():

    # usando lhs genero cosmologie (tripletti di parametri) e ci associo il pk interpolante tramite la funzione definita sopra
    
    campionatore = qmc.LatinHypercube(d=3)
    campioni_lhs = qmc.scale(campionatore.random(n=COSMOLGIE_NUM), LIMITI_INF, LIMITI_SUP)  # matrice in cui ogni riga è una cosmologia definita dai parametri nelle 3 colonne

    grid_pk = []
    grid_labels = []

    for i in tqdm(range(COSMOLGIE_NUM), desc="Integrazione CAMB"):
        om_corrente = campioni_lhs[i, 0]
        ns_corrente = campioni_lhs[i, 1]
        s8_corrente = campioni_lhs[i, 2]
        
        pk_funz = genera_pk_cosmologico(om_corrente, ns_corrente, s8_corrente)
        grid_pk.append(pk_funz)
        
        grid_labels.append([om_corrente, ns_corrente, s8_corrente])
        


    # genero campi log-normali a partire dai pk

    campioni_x = []
    labels_y = []
    pk_z = []

    for i in tqdm(range(UNIVERSI_NUM), desc="Generazione Powerbox"):
        index = np.random.randint(0, COSMOLGIE_NUM)
        label_chosen = grid_labels[index]
        pk_chosen = grid_pk[index]
        

        # USARE DIRETTAMENTE LOGNORMALPOWERBOX RESTITUISCE NAN DOVUTI A LOGARITMI DI VALORI NEGATIVI (guarda appunti su quad potsdam)
        pb = pbox.PowerBox(
            N=RESOLUTION,                     
            dim=2,                         
            pk=pk_chosen,
            boxlength=BOX_LENGTH
        )
        delta_g = pb.delta_x()  # genera campo gaussiano usando powerbox appena configurato sopra
        
        varianza = np.var(delta_g)
        delta_ln = np.exp(delta_g - varianza / 2.0) - 1.0   # genera campo ln assicurando che abbia media 0 e che non ci siano NaN o inf (derivazione formula quad potsdam)

        P_k_1D, k_bins, *extra = pbox.get_power(delta_ln, boxlength=BOX_LENGTH)
        mask = k_bins > 0
        P_k_1D_pulito = P_k_1D[mask] * BOX_LENGTH
  
        tensore_x = torch.tensor(delta_ln, dtype=torch.float32).unsqueeze(0)    # unsqueeze per aggiungere canale (1,RES,RES)
        tensore_y = torch.tensor(label_chosen, dtype=torch.float32)
        tensore_pk = torch.tensor(P_k_1D_pulito, dtype=torch.float32)
        
        campioni_x.append(tensore_x)
        labels_y.append(tensore_y)
        pk_z.append(tensore_pk)



    print("\n Salvataggio")
    os.makedirs(os.path.dirname(PERCORSO_X), exist_ok=True)
    
    tensore_x_finale = torch.stack(campioni_x)
    tensore_y_finale = torch.stack(labels_y)    # rende lista di tensori un vero tensore da usare per torch
    tensore_pk_finale = torch.stack(pk_z)
    
    torch.save(tensore_x_finale, PERCORSO_X)
    torch.save(tensore_y_finale, PERCORSO_Y)
    torch.save(tensore_pk_finale, PERCORSO_Z)
    
    print(f"Generazione completata")
    print(f"Dimensioni campi: {tensore_x_finale.shape}")
    print(f"Dimensioni labels: {tensore_y_finale.shape}")
    print(f"Dimensioni P(k) (Z): {tensore_pk_finale.shape}")

if __name__ == "__main__":
    main()
