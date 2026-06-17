#include "class_linked_list.hpp"
#include <iostream>

int main()
{
    linked_list<double> c(3.);
    c.append(5.);
    c.append(6.);
    // c.drop_first_item();
    double a = 10.;
    c.insert_next_item(10.);

    std::cout << "linked_list c= " << std::endl;
    print(c);
    std::cout << std::endl;
    std::cout << "c.length()= " << c.length() << std::endl;

    // test of operator=
    // 1st test : left list is shorter
    linked_list<double> d(15.);
    d = c;
    std::cout << "liste d (initially shorter) =" << std::endl;
    print(d);
    std::cout << std::endl;

    // 2nd test : right list is shorter
    linked_list<double> e(15.);
    d = e;
    std::cout << "liste d (initially longer)= " << std::endl;
    print(d);
    std::cout << std::endl;

    // c.truncate_items(5.);
    // std::cout << "liste c après truncate_items(5):" << std::endl;
    // print(c);

    linked_list<double> l1(1.);
    l1.append(3.);
    l1.append(5.);
    l1.append(7.);
    std::cout << "liste l1 = " << std::endl;
    print(l1);
    std::cout << std::endl;

    linked_list<double> l3(l1);
    linked_list<double> l4(l1);

    linked_list<double> l2(2.);
    l2.append(4.);
    l2.append(6.);

    std::cout << "liste l2 = " << std::endl;
    print(l2);
    std::cout << std::endl;

    l1 += l2;
    std::cout << "merge l1/ l2:" << std::endl;
    print(l1);
    std::cout << std::endl;

    // test with doublons
    l2.insert_first_item(1.);
    l2.append(7.);
    std::cout << "liste l2 modifiée:" << std::endl;
    print(l2);
    std::cout << std::endl;

    l3 += l2;
    std::cout << "merge l3/ l2:" << std::endl;
    print(l3);
    std::cout << std::endl;

    l2.append(100.);
    l2.append(101.);
    l2.append(102.);

    std::cout << "liste l2 modifiée:" << std::endl;
    print(l2);
    l4 += l2;
    std::cout << "merge l4/ l2:" << std::endl;
    print(l4);
    std::cout << std::endl;

    // test ordering algo
    linked_list<double> l5(10.);
    l5.append(9.);
    l5.append(8.);
    l5.append(7.);
    l5.append(6.);

    std::cout << "liste l5 non triée:" << std::endl;
    print(l5);
    std::cout << std::endl;

    l5.order(l5.length());
    std::cout << "liste l5 triée:" << std::endl;
    print(l5);
    std::cout << std::endl;

    // linked_list<double> l6(10.);
    // l6.append(11.);
    // l6.append(8.);
    // l6.append(100.);

    // std::cout << "liste l6 non triée:" << std::endl;
    // print(l6);
    // std::cout << std::endl;

    // l6.order(l6.length());
    
    // std::cout << "liste l6 triée:" << std::endl;
    // print(l6);
    // std::cout << std::endl;

    return 0;
}
