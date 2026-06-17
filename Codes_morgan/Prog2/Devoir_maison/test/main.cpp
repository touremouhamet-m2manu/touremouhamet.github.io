#include <iostream>
#include "class_linked_list.hpp"
#include "class_dynamic_vector.hpp"
#include "element1.hpp"
#include "row1.hpp"
#include "sparse1.hpp"

int main()
{
    std::cout << "===== Test Exercice 1 : row_element =====\n";
    {
        row_element<double> e1(3.5, 2);
        row_element<double> e2(1.5, 2);

        std::cout << "e1 = " << e1 << "\n";
        std::cout << "e2 = " << e2 << "\n";

        e1 += e2;
        std::cout << "e1 += e2 -> " << e1 << "\n";

        auto e3 = e1 * 2.0;
        std::cout << "e3 = e1 * 2 -> " << e3 << "\n";
    }

    std::cout << "\n===== Test Exercice 2 : row =====\n";
    {
        row<double> r;
        r.append(10.0, 0);
        r.append(5.0, 3);
        r.append(-2.0, 5);

        std::cout << "r[0] = " << r[0] << "\n";
        std::cout << "r[1] = " << r[1] << "\n";
        std::cout << "r[3] = " << r[3] << "\n";
        std::cout << "r[5] = " << r[5] << "\n";

        std::cout << "row_sum = " << r.row_sum() << "\n";

        r *= 2.0;
        std::cout << "apres r *= 2.0 :\n";
        std::cout << "r[0] = " << r[0] << "\n";
        std::cout << "r[3] = " << r[3] << "\n";
        std::cout << "r[5] = " << r[5] << "\n";

        dynamic_vector<double> v(6, 0.0);
        for (size_t i = 0; i < 6; ++i)
            v(i) = (double)(i + 1);   // v = (1,2,3,4,5,6)

        double prod = r * v;
        std::cout << "r * v = " << prod << "\n";
    }

    std::cout << "\n===== Test Exercice 3 : sparse_matrix =====\n";
    {
        // matrice diagonale 3x3 : 2 sur la diagonale
        sparse_matrix<double> M(3, 2.0);

        std::cout << "M(0,0) = " << M(0,0) << "\n";
        std::cout << "M(1,1) = " << M(1,1) << "\n";
        std::cout << "M(2,2) = " << M(2,2) << "\n";
        std::cout << "M(0,1) = " << M(0,1) << " (attendu 0)\n";

        dynamic_vector<double> x(3, 0.0);
        x(0) = 1.0; x(1) = 2.0; x(2) = 3.0;

        auto y = M * x;
        std::cout << "y = M * x :\n";
        std::cout << "y[0] = " << y[0] << "\n";
        std::cout << "y[1] = " << y[1] << "\n";
        std::cout << "y[2] = " << y[2] << "\n";

        std::cout << "\nDiagonale(M) :\n";
        auto D = diagonal(M);
        print(D);

        std::cout << "\nTranspose(M) :\n";
        auto Mt = transpose(M);
        print(Mt);
    }

    return 0;
}
