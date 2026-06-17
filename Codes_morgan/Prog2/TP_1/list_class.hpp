#include <iostream>
using namespace std;

template<typename T>
class list
{
protected:
    size_t number_;
    T** item_;

public:
    // --- Constructeur simple ---
    list(size_t number = 0)
        : number_(number), item_(number ? new T*[number] : nullptr)
    {
        for (size_t i = 0; i < number_; ++i)
            item_[i] = nullptr;
    }

    // --- Constructeur avec valeur initiale ---
    list(size_t number, T t)
        : number_(number), item_(number ? new T*[number] : nullptr)
    {
        for (size_t i = 0; i < number_; ++i)
            item_[i] = new T(t);
    }

    // --- Constructeur par copie ---
    list(list const& l1)
    {
        if (l1.number()) {
            number_ = l1.number();
            item_ = new T*[number_];
            for (size_t i = 0; i < number_; ++i)
                item_[i] = new T(l1[i]);
        } else {
            number_ = 0;
            item_ = nullptr;
        }
    }

    // --- Opérateur d’affectation ---
    list<T>& operator=(list const& l1)
    {
        if (this != &l1) {
            // Libération ancienne mémoire
            if (item_ != nullptr) {
                for (size_t i = 0; i < number_; ++i)
                    delete item_[i];
                delete[] item_;
            }

            number_ = l1.number();
            item_ = new T*[number_];
            for (size_t i = 0; i < number_; ++i)
                item_[i] = new T(l1[i]);
        }
        return *this;
    }

    // --- Destructeur ---
    ~list()
    {
        for (size_t i = 0; i < number_; ++i)
            delete item_[i];
        delete[] item_;
    }

    // --- Accès lecture/écriture ---
    const T& operator[](size_t ii) const { return *item_[ii]; }
    T& operator()(size_t ii) { return *item_[ii]; }

    // --- Accès aux membres ---
    size_t number() const { return number_; }
    size_t& number() { return number_; }

    const T* item(size_t ii) const { return item_[ii]; }
    T*& item(size_t ii) { return item_[ii]; }

    // --- Affichage ---
    template<typename S>
    friend ostream& operator<<(ostream& os, list<S> const& L);
};

// --- Définition de l’opérateur d’affichage ---
template<typename S>
ostream& operator<<(ostream& os, list<S> const& L)
{
    os << "[ ";
    for (size_t i = 0; i < L.number_; ++i) {
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

