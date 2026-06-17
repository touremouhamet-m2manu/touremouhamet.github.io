#pragma once
#include <iostream>

using namespace std;

template <typename T>
class linked_list
{
protected:
    T item_;
    linked_list* p_next_;

public:
    linked_list() { p_next_ = nullptr; };
    linked_list(const T& t, linked_list* N = nullptr) : item_(t), p_next_(N) {};
    linked_list(const linked_list& L) : item_(L.item()), p_next_(L.p_next() ? new linked_list(*L.p_next()) : nullptr) {};

    ~linked_list(){delete p_next_; p_next_ = nullptr;};

    linked_list* p_next() const { return p_next_; };
    linked_list*& p_next() { return p_next_; };
    const T& item() const { return item_; };
    T& item() { return item_; };

    linked_list& last() { return p_next_ ? p_next_->last() : *this; };
    size_t length() const { return p_next_ ? p_next_->length() + 1 : 1; };

    void append(const T& t) { last().p_next_ = new linked_list(t); };

    void insert_next_item(T t);
    void insert_first_item(T t);

    const linked_list<T>& operator=(const linked_list<T>& l);

    void drop_next_item();
    void drop_first_item();

    void truncate_items(double threshold);
    const linked_list<T>& operator+=(linked_list<T>& L);

    linked_list<T>& order(size_t length);
};

template <class T>
void linked_list<T>::truncate_items(double threshold)
{
    if (p_next())
    {
        if (abs(p_next()->item()) <= threshold)
        {
            drop_next_item();
            truncate_items(threshold);
        }
        else
        {
            p_next()->truncate_items(threshold);
        }
    }

    if (p_next() && abs(item()) <= threshold)
    {
        drop_first_item();
    }
}

template <class T>
void linked_list<T>::drop_next_item()
{
    if (p_next())
    {
        if (p_next()->p_next())
        {
            linked_list<T>* keep = p_next();
            p_next()              = p_next()->p_next();
            keep->item().~T();
        }
        else
        {
            delete p_next();
            p_next() = nullptr;
        }
    }
    else
    {
        std::cout << "can't remove next item: there is no next item" << std::endl;
    }
}

template <class T>
void linked_list<T>::drop_first_item()
{
    if (p_next())
    {
        item() = p_next()->item();
        drop_next_item();
    }
    else
    {
        std::cout << "can't remove first item: there is only one item" << std::endl;
    }
}

template <class T>
const linked_list<T>& linked_list<T>::operator=(const linked_list<T>& L)
{
    if (this != &L)
    {
        item() = L.item();
        if (p_next())
        {
            if (L.p_next())
            {
                *p_next() = *L.p_next();
            }
            else
            {
                delete p_next();
                p_next() = nullptr;
            }
        }
        else
        {
            if (L.p_next())
                p_next() = new linked_list(*L.p_next());
        }
    }

    return *this;
}

template <class T>
void print(const linked_list<T>& l)
{
    std::cout << "item in linked_list: " << std::endl;

    std::cout << l.item() << std::endl;

    if (l.p_next())
    {
        print(*l.p_next());
    }
} //  print a linked_list

template <typename T>
void linked_list<T>::insert_first_item(T t)
{
    p_next() = new linked_list<T>(item(), p_next());
    item() = t;
}

template <typename T>
void linked_list<T>::insert_next_item(T t)
{
    p_next() = new linked_list<T>(t, p_next());
}

template <class T>
const linked_list<T>& linked_list<T>::operator+=(linked_list<T>& L)
{
    linked_list<T>* scan1 = this;
    linked_list<T>* scan2 = &L;

    if (L.item() < item())
    {
        insert_first_item(L.item());
        scan2 = L.p_next();
    }

    for (; scan1->p_next(); scan1 = scan1->p_next())
    {
        if (scan2 && scan2->item() == scan1->item())
        {

            scan2 = scan2->p_next();
        }

        for (; scan2 && (scan2->item() < scan1->p_next()->item()); scan2 = scan2->p_next())
        {
            scan1->insert_next_item(scan2->item());
            scan1 = scan1->p_next();
        }
    }

    if (scan2 && scan2->item() == scan1->item())
    {
        
        scan2 = scan2->p_next();
    }

    if (scan2)
    {
        scan1->p_next() = new linked_list<T>(*scan2);
    }

    return *this;
}

template <class T>
linked_list<T>& linked_list<T>::order(size_t length)
{
    if (length > 1)
    {
        linked_list<T>* runner = this;
        for (auto i = 0; i < length / 2 - 1; i++)
        {
            runner = runner->p_next();
        }
        linked_list<T>* second = runner->p_next();
        runner->p_next()        = nullptr;

        order(length / 2);
        *this += second->order(length - length / 2);
    }
    return *this;
} //  order a disordered linked_list
