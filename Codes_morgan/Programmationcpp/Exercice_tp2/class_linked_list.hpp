#pragma once
#include <iostream>

template<class T>
class linked_list
{
protected:
    T item_;                     // valeur stockée
    linked_list* p_next_;        // pointeur vers le reste de la liste

public:
    // --- ACCESSEURS ---
    inline T const& item() const { return item_; }
    inline T& item() { return item_; }

    inline linked_list* p_next() const { return p_next_; }
    inline linked_list*& p_next() { return p_next_; }
    linked_list<T>& operator+=(linked_list<T> const& other);


    // --- CONSTRUCTEURS ---
    linked_list() : p_next_(nullptr) {}
    linked_list<T>& order(size_t n);    // tri

    linked_list(T const& t, linked_list* pN = nullptr)
        : item_(t), p_next_(pN) {}

    // copie récursive
    linked_list(linked_list const& L)
        : item_(L.item()),
          p_next_(L.p_next() ? new linked_list(*L.p_next()) : nullptr)
    {}

    // --- DESTRUCTEUR ---
    ~linked_list() {
        delete p_next_;      // destruction récursive automatique
        p_next_ = nullptr;
    }

    // --- FONCTIONS DE BASE ---
    linked_list& last() {
        return p_next_ ? p_next_->last() : *this;
    }

    size_t length() const {
        return p_next_ ? 1 + p_next_->length() : 1;
    }

    void append(T const& t) {
        last().p_next() = new linked_list(t);
    }

    // insère après le premier élément
    void insert_next_item(T const& t) {
        p_next_ = new linked_list(t, p_next_);
    }

    // insère avant le premier élément
    void insert_first_item(T const& t) {
        p_next_ = new linked_list(item_, p_next_);
        item_ = t;
    }

    // supprime le second élément
    void drop_next_item() {
        if (!p_next_) return;
        linked_list* tmp = p_next_;
        p_next_ = p_next_->p_next();
        tmp->p_next_ = nullptr;
        delete tmp;
    }

    // supprime le premier élément
    void drop_first_item() {
        if (!p_next_) {
            std::cout << "warning : cannot drop last item\n";
            return;
        }
        item_ = p_next_->item();
        drop_next_item();
    }

    // --- OPÉRATEUR D'AFFECTATION ---
    const linked_list& operator=(linked_list const& L) {
        if (this != &L) {
            item_ = L.item();

            delete p_next_;
            p_next_ = L.p_next() ? new linked_list(*L.p_next()) : nullptr;
        }
        return *this;
    }

    // retire tous les éléments petits
    void truncate_items(double threshold) {
        // supprimer au milieu
        if (p_next()) {
            if (p_next()->item().get_value() < threshold)
                drop_next_item();
            else
                p_next()->truncate_items(threshold);
        }

        // supprimer le premier
        if (item().get_value() < threshold)
            drop_first_item();
    }

    // --- AFFICHAGE ---
    void print() const {
        std::cout << item_ << "\n";
        if (p_next_) p_next_->print();
    }
};
template<class T>
linked_list<T>& linked_list<T>::order(size_t n)
{
    if (n <= 1) return *this;     // rien à trier

    // 1) Récupérer début
    linked_list<T>* p1 = this;

    // 2) On avance jusqu’au milieu
    for (size_t i = 1; i < n/2; i++)
        p1 = p1->p_next_;

    // 3) Diviser la liste en deux
    linked_list<T>* p2 = p1->p_next_;
    p1->p_next_ = nullptr;        // séparation physique

    // 4) Trier récursivement les deux moitiés
    this->order(n/2);
    p2->order(n - n/2);

    // 5) Fusionner avec l’opérateur +=
    *this += *p2;

    return *this;
}

template<class T>
linked_list<T>& linked_list<T>::operator+=(linked_list<T> const& other)
{
    linked_list<T>* p1 = this;
    linked_list<T>* p2 = other.p_next_ ? new linked_list<T>(other) : nullptr;

    if (!p2) return *this;  // rien à fusionner

    // Fusion : insertion ordonnée des éléments de p2 dans *this
    while (p2)
    {
        if (p2->item_ < p1->item_)
        {
            // insertion en tête
            linked_list<T>* tmp = new linked_list<T>(p2->item_, this->p_next_);
            this->item_ = tmp->item_;
            this->p_next_ = tmp->p_next_;
        }
        else
        {
            // avancer dans p1
            while (p1->p_next_ && p1->p_next_->item_ < p2->item_)
                p1 = p1->p_next_;

            // insérer après p1
            linked_list<T>* tmp = new linked_list<T>(p2->item_, p1->p_next_);
            p1->p_next_ = tmp;
        }

        p2 = p2->p_next_;
    }

    return *this;
}

