# Tesi-Francesco
Gli esperimenti posso essere eseguiti modificando le variabili specificate nei files .yaml presenti nella cartella "conf", in base alla strategia con cui si vuole eseguire la run.
Nel file main.py specificare il "conf" file che si vuole prendere come riferimento.

Le variabili di interesse che possono essere personalizzate nei vari files sono:
  - federated: boolean, per impostare un esperimento di tipo centralizzato o federated;
  - num_clients: numero clienti nel Federated Learning;
  - num_epochs: deve essere uguale a 10 per riprodurre gli esperimenti eseguiti
  - clients_per_round: deve essere sempre uguale a num_clients, si riferisce ai clienti che il server vuole includere nel processo di aggregazione;
  - num_rounds: deve essere uguale a 30;
  - quality: "normal" se si vogliono generare clients dal dataset originale, "completeness" se si vogliono sporcare o sbilanciare i clients
  - partitioning: in base all'esperimento, indicare se il partitioning viene fatto in modalità
      1. "uniform" se tutti i clients hanno la stessa size,
      2. "split_by" seguito dal nome dell'attributo se si vuole dividere i clients in base ad una feature (le features per cui esistono le funzioni di split sono indicate dalle funzioni "split_by_xxxx" alla fine del               file data.py)
      3. "balance" se si vuole dividere il dataset in clients con percentuali di class balance diverse, per cui cambiare il valore delle percentuali per ogni classe nella lista "proportions", all'interno della funzione           "get_unbalanced_subsets";
      4. "mixed" se si vogliono generare clients sia sporchi che sbilanciati, assegnando le dirty percentages nella lista "dirty" e quelle di bilanciamento delle classi nella lista "balance", all'interno della funzione           "get_mixed_subsets";
  - dirty_percentage: percentuale di sporcizia che si vuole assegnare ai clients, indicata in decimale;
  - imputation: "standard" se si vuole imputare i missing values con 0 per numerical features e "missing" per categorical, altrimenti "mean" se si vuole fare imputation con media o moda delle features;
  - num_dirty_subsets: indicare il numero di clients sul totale che si vogliono sporcare, se si vogliono sporcare tutti si può settare a 0;
  - dataset_path: indicare il percorso del dataset che si vuole utilizzare;

In fedqual_base.yaml le variabili aggiuntive in "client_fn" beta, gamma e delta si riferiscono ai coefficienti nella formula del quality weight, rispettivamente alla completeness, balance e dimensionality; in particolare, per FedQualAvg i valori vengono settati come:
  - beta: 0.50;
  - gamma: 0.50;
  - delta: 0
    
Per FedQual, invece:
  - beta: 0.33;
  - gamma: 0.33;
  - delta: 0.33

Tutte le altre variabili sono già settate al valore utilizzato per gli esperimenti, non necessitano modifiche.

