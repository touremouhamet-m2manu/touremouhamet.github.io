// ====================================================================
//                      Programme de test
// ====================================================================
#include <iostream>
//#include "Tp1_list.hpp"
using namespace std;

int main()
{
    cout << "=== Test classe list<T> du professeur ===" << endl;

    list<int> L1(5, 10);
    cout << "L1 = " << L1 << endl;

    L1(2) = 42;
    cout << "L1 modifiee = " << L1 << endl;

    list<int> L2 = L1;
    cout << "Copie L2 = " << L2 << endl;

    list<int> L3(3, 1);
    cout << "Avant affectation, L3 = " << L3 << endl;
    L3 = L1;
    cout << "Apres affectation, L3 = " << L3 << endl;

    cout << "L3[2] = " << L3[2] << endl;
    cout << "Nombre d’elements = " << L3.number() << endl;

    cout << "=== Fin du test ===" << endl;
    return 0;
}