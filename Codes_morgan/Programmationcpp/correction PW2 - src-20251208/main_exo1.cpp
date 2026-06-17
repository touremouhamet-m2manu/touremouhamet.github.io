// // ==============================================
// // FONCTION MAIN POUR TESTS
// // ==============================================
// int main() {
//     // Test des nouvelles fonctionnalités
    
//     // Création d'une liste
//     linked_list<int> list1(1);
//     list1.append(2);
//     list1.append(3);
//     list1.append(2);
//     list1.append(4);
    
//     std::cout << "Liste originale: " << std::endl;
//     print(list1);
    
//     // Test recherche
//     auto found = list1.find(3);
//     if (found) {
//         std::cout << "Element 3 trouve: " << found->item() << std::endl;
//     }
    
//     // Test suppression
//     std::cout << "\nApres suppression de tous les 2: " << std::endl;
//     list1.remove_all(2);
//     print(list1);
    
//     // Test inversion
//     std::cout << "\nApres inversion: " << std::endl;
//     list1.reverse();
//     print(list1);
    
//     // Test concaténation
//     linked_list<int> list2(5);
//     list2.append(6);
//     list1.concat(list2);
//     std::cout << "\nApres concatenation avec [5,6]: " << std::endl;
//     print(list1);
    
//     return 0;
// }
#include "exo2_linked.hpp"
#include <iostream>

// Pour tester truncate_items, créons une classe simple avec get_value()
class TestItem {
private:
    double value_;
public:
    TestItem(double v) : value_(v) {}
    double get_value() const { return value_; }
    friend std::ostream& operator<<(std::ostream& os, const TestItem& item) {
        os << item.value_;
        return os;
    }
    bool operator<(const TestItem& other) const { return value_ < other.value_; }
    bool operator==(const TestItem& other) const { return value_ == other.value_; }
};

int main() {
    std::cout << "=== TEST DE LA CLASSE linked_list ===" << std::endl;
    
    // Test 1 : Liste d'entiers
    std::cout << "\n--- Test avec des entiers ---" << std::endl;
    linked_list<int> list1(1);
    list1.append(2);
    list1.append(3);
    list1.append(2);
    list1.append(4);
    
    std::cout << "Liste originale: ";
    print(list1);
    
    std::cout << "Longueur: " << list1.length() << std::endl;
    
    // Test 2 : Recherche
    std::cout << "\n--- Recherche ---" << std::endl;
    auto found = list1.find(3);
    if (found) {
        std::cout << "Element 3 trouve" << std::endl;
    }
    
    // Test 3 : Suppression
    std::cout << "\n--- Suppression de tous les 2 ---" << std::endl;
    list1.remove_all(2);
    std::cout << "Apres suppression: ";
    print(list1);
    
    // Test 4 : Inversion
    std::cout << "\n--- Inversion ---" << std::endl;
    list1.reverse();
    std::cout << "Apres inversion: ";
    print(list1);
    
    // Test 5 : Insertions
    std::cout << "\n--- Insertions ---" << std::endl;
    list1.insert_next_item(99);
    std::cout << "Apres insert_next_item(99): ";
    print(list1);
    
    list1.insert_first_item(0);
    std::cout << "Apres insert_first_item(0): ";
    print(list1);
    
    // Test 6 : Concaténation
    std::cout << "\n--- Concatenation ---" << std::endl;
    linked_list<int> list2(5);
    list2.append(6);
    list2.append(7);
    
    std::cout << "Liste 2: ";
    print(list2);
    
    list1.concat(list2);
    std::cout << "Liste 1 apres concatenation: ";
    print(list1);
    
    // Test 7 : Copie
    std::cout << "\n--- Test de copie ---" << std::endl;
    linked_list<int> list3 = list1;  // Constructeur par copie
    std::cout << "Liste 3 (copie de liste 1): ";
    print(list3);
    
    // Test 8 : Tri (nécessite que la liste soit d'abord désordonnée)
    std::cout << "\n--- Test de tri ---" << std::endl;
    linked_list<int> list4(3);
    list4.append(1);
    list4.append(4);
    list4.append(2);
    list4.append(5);
    
    std::cout << "Liste 4 avant tri: ";
    print(list4);
    
    list4.order(list4.length());
    std::cout << "Liste 4 apres tri: ";
    print(list4);
    
    // Test 9 : Fusion de listes triées
    std::cout << "\n--- Fusion de listes triées ---" << std::endl;
    linked_list<int> list5(1);
    list5.append(3);
    list5.append(5);
    
    linked_list<int> list6(2);
    list6.append(4);
    list6.append(6);
    
    std::cout << "Liste 5: ";
    print(list5);
    std::cout << "Liste 6: ";
    print(list6);
    
    list5 += list6;
    std::cout << "Liste 5 apres fusion: ";
    print(list5);
    
    // Test 10 : Avec TestItem
    std::cout << "\n--- Test avec TestItem ---" << std::endl;
    linked_list<TestItem> list7(TestItem(2.5));
    list7.append(TestItem(1.2));
    list7.append(TestItem(3.7));
    
    std::cout << "Liste 7: ";
    print(list7);
    
    // Note: truncate_items nécessite get_value()
    // list7.truncate_items(2.0); // Décommenter si besoin
    
    return 0;
}