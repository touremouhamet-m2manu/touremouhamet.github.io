#include <iostream>
#include <cmath>   // pour pow() et abs()
using namespace std;

// On suppose que la classe template list<T> est déjà définie ici.

// ===========================================================
// 1️⃣ Fonction pour calculer les dérivées de f(x)=1/x
// ===========================================================
template<typename T>
const list<T> derive_inverse(T const& a, int n)
{
    list<T> result(n + 1); // on crée une liste de n+1 dérivées
    for (int k = 0; k <= n; ++k)
    {
        // f^(k)(a) = (-1)^k * k! / a^(k+1)
        T deriv = pow(-1, k) * tgamma(k + 1) / pow(a, k + 1);
        result.item(k) = new T(deriv);
    }
    return result;
}

// ===========================================================
// 2️⃣ Fonction pour calculer la somme de Taylor
// ===========================================================
template<class T>
const T Taylor(list<T> const& f, T const& h)
{
    T sum = 0;
    for (size_t i = 0; i < f.number(); ++i)
    {
        // f[i] * h^i / i!
        sum += f[i] * pow(h, i) / tgamma(i + 1);
    }
    return sum;
}

// ===========================================================
// 3️⃣ Programme de test
// ===========================================================
int main()
{
    cout << "=== Exercice 2 : Taylor expansion ===" << endl;

    double a = 2.0;   // point où on évalue les dérivées
    int n = 5;        // ordre de Taylor
    double h = 0.1;   // petit pas

    // On calcule les dérivées de f(x) = 1/x en a
    list<double> derivs = derive_inverse(a, n);

    cout << "Dérivées de f(x)=1/x en a=" << a << " jusqu'à l'ordre " << n << " :" << endl;
    cout << derivs << endl;

    // Calcul du développement de Taylor
    double taylor_approx = Taylor(derivs, h);
    double valeur_exacte = 1.0 / (a + h);

    cout << "\nApproximation de f(a+h) = f(" << a + h << ") :" << endl;
    cout << "Valeur approchée (Taylor) = " << taylor_approx << endl;
    cout << "Valeur exacte             = " << valeur_exacte << endl;
    cout << "Erreur absolue            = " << fabs(valeur_exacte - taylor_approx) << endl;

    cout << "=== Fin de l'exercice 2 ===" << endl;
    return 0;
}
