#include<iostream>
using namespace std;

#include "class_list.hpp"
#include"factoriel.hpp"

int main()
{
    factorial_table<long double> F(10);
    cout << "7! = " << F(7) << endl;
    cout << "10! = " << F(10) << endl;
    return 0;
}
