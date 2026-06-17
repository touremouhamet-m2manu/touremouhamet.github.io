#pragma once

template<class T>
class linked_list
{
protected:
  T item_;
  linked_list* p_next_;

public:
// ici je veux juste utiliser count au lieu de length comme l'exo du cours
// size_t length() const&
// c'est la methode de compte(value)
size_t count(T const& x)


};

template<class T>
size_t linked_list<T>::count(T const& x)
{
    return (item_ == x ? 1 : 0) + (p_next ? p_next_ ->count(x):0);
}

// ecrire remove_value(x) pour supprimer une valeur x

template<class T>
void linked_list<T>::remove_value(T const& x)
{
    while(item_ == x && p_next_) drop_first_item_();
    linked_list<T>* p= this;
    while(p->p_next())
    {
        if(p->p_next()->item()==x) 
            p->drop_next_item()
        else
        p=p->p_next();
    }
}

// La on supose que la liste est triée
// et on doit inserer t a la bonne place

template<class T>
void linked_list<T>::insert_sorted(T const& t)
{
    if(t<item()) { insert_first_item(t); return;}

    linked_list<T>* p = this;
    while(p->p_next() && p->p_next()->item()<t)
        p = p->p_next();

    p->insert_next_item(t);
}

// ici on doit implementer une somme


template<class T>
T linked_list<T>::sum() const
{
    return item_ + (p_next_ ? p_next_ -> sum() : 0);
}
