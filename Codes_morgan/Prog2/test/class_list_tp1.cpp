#include <iostream>
#include "class_list.hpp" // Supposons que tu as mis ta classe list dans ce fichier
#include "class_dynamic_vector.hpp"
#include "class_binomial.hpp" // Supposons que cette classe est fournie

// Déclaration des fonctions pour les exercices 2 et 4
template<typename T>
const list<T> derive_inverse(T const& a, int n);

template<class T>
const T Taylor(list<T> const& f, T const& h);

template<class T>
T derive_product(const list<T>& f, const list<T>& g, int n);

int main() {
    // ===== Exercice 1 : Test de la classe list =====
    std::cout << "=== Test de la classe list ===" << std::endl;

    // Test du constructeur avec allocation
    list<int> l1(5);
    std::cout << "Liste l1 (5 éléments non initialisés) : ";
    for (int i = 0; i < l1.number(); ++i) {
        l1(i) = i * 10; // Initialisation via operator()
    }
    std::cout << l1 << std::endl;
    

    // Test du constructeur avec initialisation
    list<double> l2(3, 3.14);
    std::cout << "Liste l2 (3 éléments initialisés à 3.14) : " << l2 << std::endl;

    // Test du constructeur de copie
    list<double> l3 = l2;
    std::cout << "Liste l3 (copie de l2) : " << l3 << std::endl;

    // Test de l'opérateur d'affectation
    list<double> l4(2, 2.71);
    l4 = l3;
    std::cout << "Liste l4 (après affectation de l3) : " << l4 << std::endl;

    // Test de l'accès en lecture seule et lecture/écriture
    std::cout << "Élément 1 de l4 : " << l4[1] << std::endl;
    l4(1) = 9.99;
    std::cout << "Élément 1 de l4 après modification : " << l4[1] << std::endl;

    // ===== Exercice 2 : Test de derive_inverse et Taylor =====
    std::cout << "\n=== Test de derive_inverse et Taylor ===" << std::endl;

    double a = 2.0;
    int n = 5;
    list<double> derivatives = derive_inverse(a, n);
    std::cout << "Dérivées de 1/x en x=" << a << " jusqu'à l'ordre " << n << " : " << derivatives << std::endl;

    double h = 0.1;
    double approx = Taylor(derivatives, h);
    double real_value = 1.0 / (a + h);
    std::cout << "Approximation de Taylor de 1/(" << a << "+" << h << ") : " << approx << std::endl;
    std::cout << "Valeur réelle : " << real_value << std::endl;

    // ===== Exercice 3 : Test de la classe binomial =====
    std::cout << "\n=== Test de la classe binomial ===" << std::endl;

    binomial<int> pascal(5); // Triangle de Pascal jusqu'à la ligne 5
    std::cout << "Coefficients binomiaux pour n=2 : ";
    for (int k = 0; k <= 2; ++k) {
        std::cout << pascal(2, k) << " ";
    }
    std::cout << std::endl;

    std::cout << "Coefficients binomiaux pour n=4 : ";
    for (int k = 0; k <= 4; ++k) {
        std::cout << pascal(4, k) << " ";
    }
    std::endl;

    // ===== Exercice 4 : Test de derive_product =====
    std::cout << "\n=== Test de derive_product ===" << std::endl;

    // Exemple : f(x) = x^2, g(x) = x^3
    // Dérivées de f : f(0)=0, f'(0)=0, f''(0)=2, f'''(0)=0, f''''(0)=0
    list<double> f_derivatives(5);
    f_derivatives(0) = 0; // f(x) = x^2
    f_derivatives(1) = 0; // f'(x) = 2x → f'(0) = 0
    f_derivatives(2) = 2; // f''(x) = 2 → f''(0) = 2
    f_derivatives(3) = 0; // f'''(x) = 0 → f'''(0) = 0
    f_derivatives(4) = 0; // f''''(x) = 0 → f''''(0) = 0

    // Dérivées de g : g(0)=0, g'(0)=0, g''(0)=0, g'''(0)=6, g''''(0)=0
    list<double> g_derivatives(5);
    g_derivatives(0) = 0; // g(x) = x^3
    g_derivatives(1) = 0; // g'(x) = 3x^2 → g'(0) = 0
    g_derivatives(2) = 0; // g''(x) = 6x → g''(0) = 0
    g_derivatives(3) = 6; // g'''(x) = 6 → g'''(0) = 6
    g_derivatives(4) = 0; // g''''(x) = 0 → g''''(0) = 0

    int order = 4;
    double product_derivative = derive_product(f_derivatives, g_derivatives, order);
    std::cout << "Dérivée d'ordre " << order << " de (x^2 * x^3) en x=0 : " << product_derivative << std::endl;
    // Résultat attendu : (x^5)''''(0) = 120

    return 0;
}
