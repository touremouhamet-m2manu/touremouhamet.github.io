#include <iostream>
#include <iomanip>  // pour un affichage propre
using namespace std;

/**
 * Classe template list<T>
 * -----------------------
 * Cette classe représente une liste dynamique d'objets de type T
 * stockés sous forme de pointeurs (T*).
 * Elle permet la gestion mémoire, la copie, et la manipulation
 * d’éléments via des opérateurs surchargés.
 */

template <typename T>
class list {
protected:
    int number_;   // nombre d'éléments dans la liste
    T** item_;     // tableau de pointeurs vers les objets T

public:
    // --- (1) Constructeur de base : alloue un tableau de pointeurs vides ---
    list(int number = 0)
        : number_(number)
    {
        if (number_ > 0)
            item_ = new T*[number_];
        else
            item_ = nullptr;

        for (int i = 0; i < number_; ++i)
            item_[i] = nullptr;
    }

    // --- (2) Constructeur avec valeur initiale ---
    list(int number, T const& value)
        : number_(number)
    {
        item_ = new T*[number_];
        for (int i = 0; i < number_; ++i)
            item_[i] = new T(value);
    }

    // --- (3) Constructeur par copie ---
    list(list<T> const& other)
        : number_(other.number_)
    {
        item_ = new T*[number_];
        for (int i = 0; i < number_; ++i)
            item_[i] = new T(*(other.item_[i]));
    }

    // --- (4) Opérateur d’affectation ---
    list<T>& operator=(list<T> const& other)
    {
        if (this != &other) {
            // libère la mémoire existante
            for (int i = 0; i < number_; ++i)
                delete item_[i];
            delete[] item_;

            // copie
            number_ = other.number_;
            item_ = new T*[number_];
            for (int i = 0; i < number_; ++i)
                item_[i] = new T(*(other.item_[i]));
        }
        return *this;
    }

    // --- (5) Destructeur ---
    ~list()
    {
        for (int i = 0; i < number_; ++i)
            delete item_[i];
        delete[] item_;
    }

    // --- (6) Accès en lecture ---
    const T& operator[](int i) const
    {
        if (i < 0 || i >= number_)
            throw out_of_range("Index out of range (lecture)");
        return *(item_[i]);
    }

    // --- (7) Accès en écriture ---
    T& operator()(int i)
    {
        if (i < 0 || i >= number_)
            throw out_of_range("Index out of range (écriture)");
        return *(item_[i]);
    }

    // --- (8) Accès à number_ ---
    int number() const { return number_; }
    int& number() { return number_; }

    // --- (9) Accès à item_[i] ---
    const T* item(int i) const { return item_[i]; }
    T*& item(int i) { return item_[i]; }

    // --- (10) Affichage de la liste ---
    friend ostream& operator<<(ostream& os, list<T> const& L)
    {
        os << "[ ";
        for (int i = 0; i < L.number_; ++i) {
            if (L.item_[i])
                os << *(L.item_[i]);
            else
                os << "null";
            if (i < L.number_ - 1)
                os << ", ";
        }
        os << " ]";
        return os;
    }
};

// ============================================================================
//                           Programme de test
// ============================================================================

int main() {
    cout << "=== Test de la classe template list<T> ===" << endl;

    // Création d’une liste de 5 entiers initialisés à 10
    list<int> L1(5, 10);
    cout << "L1 = " << L1 << endl;

    // Modification d’un élément
    L1(2) = 42;
    cout << "L1 modifiée = " << L1 << endl;

    // Test du constructeur par copie
    list<int> L2 = L1;
    cout << "Copie L2 = " << L2 << endl;

    // Test de l’opérateur d’affectation
    list<int> L3(3, 1);
    cout << "Avant affectation, L3 = " << L3 << endl;
    L3 = L1;
    cout << "Après affectation, L3 = " << L3 << endl;

    // Lecture d’un élément via []
    cout << "L3[2] = " << L3[2] << endl;

    // Vérification de la taille
    cout << "Nombre d’éléments dans L3 = " << L3.number() << endl;

    cout << "=== Fin du test ===" << endl;
    return 0;
}
