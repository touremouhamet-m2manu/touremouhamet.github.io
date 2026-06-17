#include <iostream>
#include <stdexcept>

template<typename T>
class list {
protected:
    int number_;
    T** item_;

public:
    // Constructeur avec allocation
    list(int number) : number_(number) {
        item_ = new T*[number_];
        for (int i = 0; i < number_; ++i) {
            item_[i] = nullptr;
        }
    }

    // Constructeur avec initialisation
    list(int number, const T& init) : number_(number) {
        item_ = new T*[number_];
        for (int i = 0; i < number_; ++i) {
            item_[i] = new T(init);
        }
    }

    // Constructeur de copie
    list(const list<T>& other) : number_(other.number_) {
        item_ = new T*[number_];
        for (int i = 0; i < number_; ++i) {
            item_[i] = new T(*other.item_[i]);
        }
    }

    // Opérateur d'affectation
    list<T>& operator=(const list<T>& other) {
        if (this != &other) {
            for (int i = 0; i < number_; ++i) {
                delete item_[i];
            }
            delete[] item_;

            number_ = other.number_;
            item_ = new T*[number_];
            for (int i = 0; i < number_; ++i) {
                item_[i] = new T(*other.item_[i]);
            }
        }
        return *this;
    }

    // Destructeur
    ~list() {
        for (int i = 0; i < number_; ++i) {
            delete item_[i];
        }
        delete[] item_;
    }

    // Accès en lecture seule
    const T& operator[](int i) const {
        if (i < 0 || i >= number_) {
            throw std::out_of_range("Index out of range");
        }
        return *item_[i];
    }

    // Accès en lecture/écriture
    T& operator()(int i) {
        if (i < 0 || i >= number_) {
            throw std::out_of_range("Index out of range");
        }
        return *item_[i];
    }

    // Accès au nombre d'éléments
    int number() const { return number_; }

    // Accès à un élément (lecture seule)
    const T* item(int i) const {
        if (i < 0 || i >= number_) {
            throw std::out_of_range("Index out of range");
        }
        return item_[i];
    }

    // Accès à un élément (lecture/écriture)
    T* item(int i) {
        if (i < 0 || i >= number_) {
            throw std::out_of_range("Index out of range");
        }
        return item_[i];
    }

    // Affichage
    friend std::ostream& operator<<(std::ostream& os, const list<T>& l) {
        for (int i = 0; i < l.number_; ++i) {
            os << *l.item_[i] << " ";
        }
        return os;
    }
};
