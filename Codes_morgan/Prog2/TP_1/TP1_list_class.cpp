#include <iostream>
using namespace std;

// ===================== Classe list<T> =====================
template<typename T>
class list
{
protected:
    size_t number_;
    T** item_;

public:
    list(size_t number = 0)
        : number_(number), item_(number ? new T*[number] : nullptr)
    {
        for (size_t i = 0; i < number_; ++i)
            item_[i] = nullptr;
    }

    list(size_t number, T t)
        : number_(number), item_(number ? new T*[number] : nullptr)
    {
        for (size_t i = 0; i < number_; ++i)
            item_[i] = new T(t);
    }

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

    list<T>& operator=(list const& l1)
    {
        if (this != &l1) {
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

    ~list()
    {
        for (size_t i = 0; i < number_; ++i)
            delete item_[i];
        delete[] item_;
    }

    const T& operator[](size_t ii) const { return *item_[ii]; }
    T& operator()(size_t ii) { return *item_[ii]; }

    size_t number() const { return number_; }
    size_t& number() { return number_; }

    const T* item(size_t ii) const { return item_[ii]; }
    T*& item(size_t ii) { return item_[ii]; }

    template<typename S>
    friend ostream& operator<<(ostream& os, list<S> const& L);
};

// --- opérateur << ---
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

// ===================== Programme principal =====================
int main()
{
    cout << "=== Test classe list<T> ===" << endl;

    list<double> L1(5, 10);
    cout << "L1 = " << L1 << endl;

    L1(2) = 42;
    cout << "L1 modifiee = " << L1 << endl;

    list<double> L2 = L1;
    cout << "Copie L2 = " << L2 << endl;

    list<double> L3(3, 1);
    cout << "Avant affectation, L3 = " << L3 << endl;
    L3 = L1;
    cout << "Apres affectation, L3 = " << L3 << endl;

    cout << "L3[2] = " << L3[2] << endl;
    cout << "Nombre d elements = " << L3.number() << endl;

    cout << "=== Fin du test ===" << endl;
    return 0;
}
