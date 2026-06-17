template<class T>
linked_list<T>& linked_list<T>::order(size_t n)
{
    if (n <= 1) return *this;

    // découpage au milieu
    linked_list<T>* p1 = this;
    for (size_t i = 1; i < n/2; i++)
        p1 = p1->p_next();

    linked_list<T>* p2 = p1->p_next();
    p1->p_next() = nullptr;    // séparation

    // tri récursif des deux moitiés
    this->order(n/2);
    p2->order(n - n/2);

    // fusion
    *this += *p2;

    return *this;
}
