// #include "class_linked_list.hpp"

// int main()
// {
//     linked_list<double> L(5);
//     L.append(2);
//     L.append(7);
//     L.append(1);
//     L.append(9);

//     L.print();
//     std::cout << "length = " << L.length() << "\n";

//     L.order(L.length());
//     std::cout << "\nSorted :\n";
//     L.print();

//     return 0;
// }
#include "class_linked_list.hpp"
#include <iostream>

int main()
{
    linked_list<double> L(5.);   // premier élément = 5
    L.append(2.);
    L.append(7.);
    L.append(1.);
    L.append(9.);

    std::cout << "Liste initiale :\n";
    L.print();

    L.order(L.length());


    std::cout << "\nListe triée :\n";
    L.print();

    return 0;
}
