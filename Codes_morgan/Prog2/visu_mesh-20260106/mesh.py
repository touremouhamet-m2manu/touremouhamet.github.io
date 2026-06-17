import matplotlib.pyplot as plt

def plot_mesh_from_file(filename):
    # Initialisation des listes pour stocker les coordonnées
    x_coords = []
    y_coords = []
    edge_count = 0
    current_segment = []

    try:
        # Ouverture du fichier en mode lecture ('r')
        with open(filename, 'r') as f:
            for line in f:
                clean_line = line.strip()
                
                # Si la ligne est vide, on ignore et on réinitialise le segment
                if not clean_line:
                    current_segment = []
                    continue
                
                try:
                    # Séparation des deux colonnes (x et y)
                    coords = clean_line.split()
                    if len(coords) == 2:
                        current_segment.append([float(coords[0]), float(coords[1])])
                    
                    # Dès qu'on a une paire de points, on prépare le segment
                    if len(current_segment) == 2:
                        p1, p2 = current_segment[0], current_segment[1]
                        
                        # Ajout des coordonnées avec un 'None' pour séparer les segments
                        # Cela évite que matplotlib ne relie la fin du segment 1 au début du segment 2
                        x_coords.extend([p1[0], p2[0], None])
                        y_coords.extend([p1[1], p2[1], None])
                        
                        edge_count += 1
                        current_segment = [] # On vide pour le segment suivant
                except ValueError:
                    # Ignore les lignes contenant du texte au lieu de chiffres
                    continue

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{filename}' n'existe pas dans le répertoire.")
        return

    # Création du graphique
    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, color='blue', linewidth=0.8)
    
    # Paramètres d'affichage
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Maillage - {edge_count} Segments", fontsize=14)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    print(f"Succès : {edge_count} segments lus et tracés.")
    plt.show()

plot_mesh_from_file('segment.txt')