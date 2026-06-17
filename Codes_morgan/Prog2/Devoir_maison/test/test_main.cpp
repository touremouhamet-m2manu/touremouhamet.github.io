// #include "row_element1.hpp"
// #include <iostream>

// int main() {
//     row_element<double> e1(3.5, 2);
//     row_element<double> e2(1.5, 2);

//     std::cout << "e1 = " << e1 << "\n";
//     std::cout << "e2 = " << e2 << "\n";

//     e1 += e2;
//     std::cout << "e1 += e2 -> " << e1 << "\n";

//     auto e3 = e1 * 2.0;
//     std::cout << "e3 = e1 * 2 -> " << e3 << "\n";

//     std::cout << "e1 < e3 ? " << (e1 < e3) << "\n";

//     return 0;
// }

#include <iostream>
#include "row_element1.hpp"

int main() {
    // Création d'éléments
    row_element<double> e1(3.5, 2);
    row_element<double> e2(1.5, 2);
    row_element<double> e3(4.0, 5);

    // Affichage
    std::cout << "e1 = " << e1 << std::endl;
    std::cout << "e2 = " << e2 << std::endl;
    std::cout << "e3 = " << e3 << std::endl;

    // Opérateurs arithmétiques
    e1 += 2.0;
    std::cout << "e1 += 2 -> " << e1 << std::endl;

    e1 -= e2;
    std::cout << "e1 -= e2 -> " << e1 << std::endl;

    row_element<double> e4 = e1 * 3.0;
    std::cout << "e4 = e1 * 3 -> " << e4 << std::endl;

    row_element<double> e5 = 10.0 - e2;
    std::cout << "e5 = 10 - e2 -> " << e5 << std::endl;

    // Comparaisons (sur column_)
    if (e1 == e2)
        std::cout << "e1 et e2 sont dans la même colonne" << std::endl;

    if (e1 < e3)
        std::cout << "e1 est avant e3 (colonne)" << std::endl;

    return 0;
}
