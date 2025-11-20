import tensorflow as tf

# Le chemin vers le dossier d'une époque (pas le fichier, le dossier + préfixe)
# Note: On ne met pas l'extension .data ou .meta, juste le préfixe
ckpt_path = "/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/ChronoRoot/modelWeights/ResUNetDS/epoch_67/model.ckpt"

print(f"Tentative de lecture de : {ckpt_path}")

try:
    # Cette fonction permet de lister les variables dans le checkpoint
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    print("SUCCESS ! Le modèle contient les variables suivantes (extrait) :")
    count = 0
    for key in var_to_shape_map:
        print(" - tensor_name: ", key)
        count += 1
        if count > 5: break # On en affiche juste 5 pour vérifier
except Exception as e:
    print("ERREUR : Le modèle n'a pas pu être lu.")
    print(e)