// #include <iostream>
#include "row_object1.hpp"

int main() {
    // 1) Création d'une ligne vide
    row<double> r;  // aucun élément

    std::cout << "=== Test insertion ===\n";
    // On ajoute quelques éléments non nuls
    r.append(10.0, 0);   // colonne 0
    r.append(5.0, 3);    // colonne 3
    r.append(-2.0, 5);   // colonne 5

    // 2) Test de l'accès par operator[]
    std::cout << "r[0] = " << r[0] << "\n";  // doit afficher 10
    std::cout << "r[1] = " << r[1] << "\n";  // doit afficher 0 (élément absent)
    std::cout << "r[3] = " << r[3] << "\n";  // doit afficher 5
    std::cout << "r[5] = " << r[5] << "\n";  // doit afficher -2

    // 3) Somme de la ligne
    std::cout << "row_sum = " << r.row_sum() << "\n";  // doit afficher 10 + 5 - 2 = 13

    // 4) Test des opérateurs * et *= avec un scalaire
    std::cout << "\n=== Test produit par un scalaire ===\n";
    row<double> r2 = r * 2.0;  // utilise l'opérateur non-membre
    std::cout << "r2[0] = " << r2[0] << "\n"; // 20
    std::cout << "r2[3] = " << r2[3] << "\n"; // 10
    std::cout << "r2[5] = " << r2[5] << "\n"; // -4

    r *= 0.5;
    std::cout << "après r *= 0.5 :\n";
    std::cout << "r[0] = " << r[0] << "\n"; // 5
    std::cout << "r[3] = " << r[3] << "\n"; // 2.5
    std::cout << "r[5] = " << r[5] << "\n"; // -1

    // 5) Test du produit scalaire avec dynamic_vector
    std::cout << "\n=== Test produit scalaire row * dynamic_vector ===\n";
    // on suppose un constructeur dynamic_vector<double>(int n)
    dynamic_vector<double> v(6);
    // on suppose que v[i] est modifiable (operator[] renvoie T& non-const)
    v[0] = 1.0;
    v[1] = 2.0;
    v[2] = 3.0;
    v[3] = 4.0;
    v[4] = 5.0;
    v[5] = 6.0;

    double prod = r * v;  // appelle row<T>::operator*(dynamic_vector<T> const&)
    std::cout << "r * v = " << prod << "\n";

    // 6) Test de drop_items : on garde seulement colonnes 0 et 3
    std::cout << "\n=== Test drop_items ===\n";
    dynamic_vector<int> mask(6);
    mask[0] = 1;  // garder colonne 0
    mask[1] = 0;
    mask[2] = 0;
    mask[3] = 1;  // garder colonne 3
    mask[4] = 0;
    mask[5] = 0;  // supprimer colonne 5

    r.drop_items(mask);

    std::cout << "après drop_items(mask):\n";
    std::cout << "r[0] = " << r[0] << "\n"; // doit rester non nul
    std::cout << "r[3] = " << r[3] << "\n"; // doit rester non nul
    std::cout << "r[5] = " << r[5] << "\n"; // doit être 0 (supprimé)

    return 0;
}
