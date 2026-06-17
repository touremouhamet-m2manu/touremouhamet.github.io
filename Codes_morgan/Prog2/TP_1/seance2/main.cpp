int main()
{
    cout << "=== Exercice 2 : Taylor expansion ===" << endl;

    double a = 2.0;
    size_t n = 5;
    double h = 0.1;

    // Calcul des dérivées successives de 1/x en a
    list<double> derivs = derive_inverse(a, n);

    cout << "Dérivées de f(x)=1/x en a=" << a << " :" << endl;
    cout << derivs << endl;

    // Approximation de f(a+h)
    double approx = Taylor(derivs, h);
    double exact = 1.0 / (a + h);

    cout << "\nApproximation de f(a+h) = f(" << a + h << "):" << endl;
    cout << "Valeur approchée : " << approx << endl;
    cout << "Valeur exacte    : " << exact << endl;
    cout << "Erreur           : " << fabs(approx - exact) << endl;

    return 0;
}
