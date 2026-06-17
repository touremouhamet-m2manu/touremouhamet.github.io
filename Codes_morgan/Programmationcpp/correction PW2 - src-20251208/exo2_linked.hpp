#pragma once
#include <iostream>

// ==============================================
// DÉCLARATION DE LA CLASSE
// ==============================================

template <class T>
class linked_list
{
protected:
    T item_;
    linked_list* p_next_;

public:
    // Accesseurs
    inline T item() const { return item_; };
    inline T& item() { return item_; };
    inline linked_list* p_next() const { return p_next_; };
    inline linked_list*& p_next() { return p_next_; };

    // Constructeurs et destructeur
    linked_list();
    linked_list(T const& t, linked_list* pN = nullptr);
    linked_list(linked_list const& L);
    ~linked_list();
    
    // Méthodes de base
    linked_list& last();
    size_t length() const;
    void append(T const& t);
    void insert_next_item(T const& t);
    void insert_first_item(T const& t);
    const linked_list& operator=(linked_list const&);
    void drop_next_item();
    void drop_first_item();
    void truncate_items(double threshold);
    
    // Fonctions de fusion et tri
    const linked_list& operator+=(linked_list&);
    linked_list& order(size_t);
    
    // Exercices d'entraînement
    void reverse();
    void reverse_recursive();
    linked_list* find(const T& val);
    linked_list* find_recursive(const T& val);
    void remove_all(const T& val);
    void remove_all_recursive(const T& val);
    void concat(linked_list& other);
    
    // Fonction d'affichage amie
    template <class U>
    friend void print(linked_list<U> const& mylist);
};

// Déclaration de la fonction d'affichage
template <class T>
void print(linked_list<T> const& mylist);

// ==============================================
// IMPLÉMENTATION DES MÉTHODES
// ==============================================

// Constructeurs et destructeur
template <class T>
linked_list<T>::linked_list() : p_next_(nullptr) {}

template <class T>
linked_list<T>::linked_list(T const& t, linked_list* pN) : item_(t), p_next_(pN) {}

template <class T>
linked_list<T>::linked_list(linked_list const& L) : item_(L.item()), p_next_(L.p_next() ? new linked_list(*L.p_next()) : nullptr) {}

template <class T>
linked_list<T>::~linked_list() { delete p_next_; }

// Dernier élément
template <class T>
linked_list<T>& linked_list<T>::last()
{
    return p_next_ ? p_next_->last() : *this;
}

// Longueur de la liste
template <class T>
size_t linked_list<T>::length() const
{
    return p_next_ ? p_next_->length() + 1 : 1;
}

// Ajouter à la fin
template <class T>
void linked_list<T>::append(T const& t)
{
    last().p_next() = new linked_list(t);
}

// Insertion après l'élément courant
template <class T>
void linked_list<T>::insert_next_item(T const& t)
{
    p_next_ = new linked_list<T>(t, p_next_);
}

// Insertion avant le premier élément
template <class T>
void linked_list<T>::insert_first_item(T const& t)
{
    p_next_ = new linked_list<T>(item_, p_next_);
    item_ = t;
}

// Opérateur d'affectation
template <class T>
const linked_list<T>& linked_list<T>::operator=(linked_list<T> const& L)
{
    if (this != &L)
    {
        item_ = L.item_;

        if (p_next_)
        {
            if (L.p_next_)
            {
                *p_next_ = *L.p_next_;
            }
            else
            {
                delete p_next_;
                p_next_ = nullptr;
            }
        }
        else if (L.p_next_)
        {
            p_next_ = new linked_list(*L.p_next_);
        }
    }
    return *this;
}

// Supprimer l'élément suivant
template <class T>
void linked_list<T>::drop_next_item()
{
    if (p_next_)
    {
        if (p_next_->p_next_)
        {
            linked_list<T>* keep = p_next_;
            p_next_ = p_next_->p_next_;
            keep->p_next_ = nullptr;
            delete keep;
        }
        else
        {
            delete p_next_;
            p_next_ = nullptr;
        }
    }
}

// Supprimer le premier élément
template <class T>
void linked_list<T>::drop_first_item()
{
    if (p_next_)
    {
        item_ = p_next_->item_;
        drop_next_item();
    }
    else
    {
        std::cout << "warning : only one element in the list - impossible to drop" << std::endl;
    }
}

// Tronquer selon un seuil
template <class T>
void linked_list<T>::truncate_items(double threshold)
{
    if (p_next_)
    {
        // Pour tester cette fonction, on suppose que T a une méthode get_value()
        // Si ce n'est pas le cas, il faudra adapter
        if (p_next_->item_.get_value() < threshold)
        {
            drop_next_item();
            truncate_items(threshold);
        }
        else
        {
            p_next_->truncate_items(threshold);
        }
    }
    if (p_next_ && item_.get_value() < threshold)
    {
        drop_first_item();
    }
}

// Fonction d'affichage
template <class T>
void print(linked_list<T> const& mylist)
{
    std::cout << mylist.item_ << " ";
    if (mylist.p_next_)
        print(*mylist.p_next_);
    else
        std::cout << std::endl;
}

// Fusion de deux listes triées
template <class T>
const linked_list<T>& linked_list<T>::operator+=(linked_list<T>& L)
{
    linked_list<T>* p_scan1 = this;
    linked_list<T>* p_scan2 = &L;

    if (L.item_ < item_)
    {
        insert_first_item(L.item_);
        p_scan2 = p_scan2->p_next_;
    }
    for (; p_scan1->p_next_; p_scan1 = p_scan1->p_next_)
    {
        if (p_scan2 && p_scan1->item_ == p_scan2->item_)
            p_scan2 = p_scan2->p_next_;

        for (; p_scan2 && (p_scan2->item_ < p_scan1->p_next_->item_); p_scan2 = p_scan2->p_next_)
        {
            p_scan1->insert_next_item(p_scan2->item_);
            p_scan1 = p_scan1->p_next_;
        }
    }
    if (p_scan2 && p_scan1->item_ == p_scan2->item_)
        p_scan2 = p_scan2->p_next_;

    if (p_scan2)
        p_scan1->p_next_ = new linked_list(*p_scan2);

    return *this;
}

// Tri (merge sort)
template <class T>
linked_list<T>& linked_list<T>::order(size_t length)
{
    if (length > 1)
    {
        linked_list<T>* p_scan1 = this;
        for (auto count = 0; count < length / 2 - 1; ++count)
        {
            p_scan1 = p_scan1->p_next_;
        }
        linked_list<T>* p_scan2 = p_scan1->p_next_;
        p_scan1->p_next_ = nullptr;

        this->order(length / 2);
        (*this).operator+=(p_scan2->order(length - length / 2));
    }
    return *this;
}

// ==============================================
// EXERCICES D'ENTRAÎNEMENT
// ==============================================

// Inversion itérative
template <class T>
void linked_list<T>::reverse()
{
    if (!p_next_) return;
    
    linked_list<T>* prev = nullptr;
    linked_list<T>* current = this;
    linked_list<T>* next = nullptr;
    
    while (current) {
        next = current->p_next_;
        current->p_next_ = prev;
        prev = current;
        current = next;
    }
    
    if (prev) {
        T temp_item = item_;
        item_ = prev->item_;
        prev->item_ = temp_item;
        
        linked_list<T>* temp_next = p_next_;
        p_next_ = prev->p_next_;
        prev->p_next_ = temp_next;
    }
}

// Inversion récursive
template <class T>
void linked_list<T>::reverse_recursive()
{
    if (!p_next_) return;
    
    linked_list<T>* rest = p_next_;
    rest->reverse_recursive();
    
    if (rest->p_next_) {
        linked_list<T>* last = rest;
        while (last->p_next_) {
            last = last->p_next_;
        }
        last->p_next_ = new linked_list<T>(item_, nullptr);
    } else {
        rest->p_next_ = new linked_list<T>(item_, nullptr);
    }
    
    if (p_next_) {
        linked_list<T>* old_next = p_next_;
        item_ = p_next_->item_;
        p_next_ = p_next_->p_next_;
        old_next->p_next_ = nullptr;
        delete old_next;
    }
}

// Recherche itérative
template <class T>
linked_list<T>* linked_list<T>::find(const T& val)
{
    linked_list<T>* current = this;
    while (current) {
        if (current->item_ == val) {
            return current;
        }
        current = current->p_next_;
    }
    return nullptr;
}

// Recherche récursive
template <class T>
linked_list<T>* linked_list<T>::find_recursive(const T& val)
{
    if (item_ == val) {
        return this;
    }
    if (!p_next_) {
        return nullptr;
    }
    return p_next_->find_recursive(val);
}

// Suppression de tous les éléments égaux à val (itératif)
template <class T>
void linked_list<T>::remove_all(const T& val)
{
    // Suppression des éléments en tête
    while (p_next_ && item_ == val) {
        drop_first_item();
    }
    
    // Parcours du reste
    linked_list<T>* current = this;
    while (current && current->p_next_) {
        if (current->p_next_->item_ == val) {
            linked_list<T>* to_delete = current->p_next_;
            current->p_next_ = to_delete->p_next_;
            to_delete->p_next_ = nullptr;
            delete to_delete;
        } else {
            current = current->p_next_;
        }
    }
}

// Suppression de tous les éléments égaux à val (récursif)
template <class T>
void linked_list<T>::remove_all_recursive(const T& val)
{
    if (!p_next_) return;
    
    if (p_next_->item_ == val) {
        drop_next_item();
        remove_all_recursive(val);
    } else {
        p_next_->remove_all_recursive(val);
    }
    
    if (item_ == val && p_next_) {
        drop_first_item();
    }
}

// Concaténation de deux listes
template <class T>
void linked_list<T>::concat(linked_list<T>& other)
{
    linked_list<T>* last_node = &(this->last());
    if (last_node) {
        last_node->p_next_ = new linked_list<T>(other.item_, nullptr);
        
        if (other.p_next_) {
            linked_list<T>* current = last_node->p_next_;
            linked_list<T>* other_current = other.p_next_;
            
            while (other_current) {
                current->p_next_ = new linked_list<T>(other_current->item_, nullptr);
                current = current->p_next_;
                other_current = other_current->p_next_;
            }
        }
    }
}