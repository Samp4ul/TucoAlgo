import csv
import urllib.request
import os.path

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Définir le répertoire de destination pour les images téléchargées
destination_directory = "/Users/ellioth/Documents/PI2/GitHub/crop-image-becqe/ressources/images/"

with open('/Users/ellioth/Documents/PI2/GitHub/crop-image-becqe/ressources/images/batiments_irc.csv', 'r') as file:
    # Itérer à travers les lignes du fichier CSV
    cpt = 0
    for row in file:
        row = row.split(';')
        # Obtenir l'URL de l'image depuis la deuxième colonne (indice 1)
        image_url = row[1]
        # Télécharger l'image depuis l'URL
        urllib.request.urlretrieve(image_url, os.path.join(destination_directory, str(cpt)+'.jpg'))
        cpt += 1

print(f"{cpt} images téléchargées avec succès.")


