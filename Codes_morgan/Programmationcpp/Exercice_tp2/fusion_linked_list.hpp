template<class T>
const linked_list<T>& linked_list<T>::operator+=(linked_list<T>& L)
{
    linked_list<T>* p1 = this;
    linked_list<T>* p2 = &L;

    // 1) vérifier le premier élément
    if (p2 && p2->item() < p1->item()) {
        insert_first_item(p2->item());
        p2 = p2->p_next();
    }

    // 2) boucle principale
    for (; p1->p_next(); p1 = p1->p_next()) {

        // égalité -> on skip (pas de doublons)
        if (p2 && p2->item() == p1->item()) {
            p2 = p2->p_next();
        }

        // insérer les éléments de L qui doivent venir ici
        while (p2 && p2->item() < p1->p_next()->item()) {
            p1->insert_next_item(p2->item());
            p1 = p1->p_next();   // avancer
            p2 = p2->p_next();
        }
    }

    // 3) ajouter les éléments restants à la fin
    while (p2) {
        p1->append(p2->item());
        p2 = p2->p_next();
    }

    return *this;
}
