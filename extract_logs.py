import wandb
import pandas as pd

api = wandb.Api()
project_path = "lgand-universit-de-montpellier/chronoRoot_logs"

print(f"Récupération de la liste des runs pour le projet : {project_path}...")
runs = api.runs(project_path)

all_runs_data = []

for run in runs:
    print(f"Téléchargement des données pour le run : {run.name} (ID: {run.id})...")

    config = run.config
    model_name = config.get("Model", config.get("model", "N/A"))
    loss_fn = config.get("loss", "N/A")
    l2_val = config.get("l2", "N/A")
    runtime = run.summary.get("_runtime", "N/A") 
    
    history = run.scan_history()
    
    for row in history:
        row['run_name'] = run.name
        row['runtime'] = runtime
        
        row['model'] = model_name
        row['loss'] = loss_fn
        row['l2'] = l2_val
        
        all_runs_data.append(row)

print("Conversion des données en tableau (cela peut prendre un instant)...")
df = pd.DataFrame(all_runs_data)

# 3. Sauvegarder en fichier CSV
nom_fichier_csv = "log_chronoRoot.csv"
df.to_csv(nom_fichier_csv, index=False)

print(f"Succès ! L'ensemble des données a été sauvegardé et trié dans : {nom_fichier_csv}")