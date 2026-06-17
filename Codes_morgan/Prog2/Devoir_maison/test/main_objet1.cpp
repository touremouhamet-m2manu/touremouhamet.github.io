#include <iostream>
#include "row_object1.hpp"

int main() {
    row<double> r;

    r.append(2.0, 0);
    r.append(3.0, 2);
    r.append(5.0, 4);

    std::cout << "r[0] = " << r[0] << std::endl;
    std::cout << "r[1] = " << r[1] << std::endl;
    std::cout << "r[2] = " << r[2] << std::endl;

    std::cout << "Somme ligne = " << r.row_sum() << std::endl;

    r *= 2.0;
    std::cout << "Apres *2, r[2] = " << r[2] << std::endl;

    dynamic_vector<double> v(6, 1.0);
    std::cout << "Produit scalaire r·v = " << (r * v) << std::endl;

    dynamic_vector<int> mask(6, 1);
    mask[2] = 0; // supprimer colonne 2
    r.drop_items(mask);

    std::cout << "Apres drop, r[2] = " << r[2] << std::endl;

    return 0;
}