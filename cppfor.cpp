
#include <cmath>
#include <iostream>
#include <string>
//#include <boost/serialization/array_wrapper.hpp>

#include <vector>
// Optimization
#define BOOST_UBLAS_NDEBUG
#include <stdlib.h>
#include <time.h>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <exception>
#include <fstream>
#include <string>
#include <utility>

#include <algorithm>
#include <functional>
#include <utility>

extern "C" {
void ddriv2_(int* n, double* tv, double* yv,
             void(f)(int* n, double* t, double* yv, double* ydotv),
             double* tout, int* mstate, int* nroot, double* eps, double* ewt,
             int* mint, double* work, int* lenw, double* iwork, int* leniw,
             void(g)(int* n, double* t, double* yv, int* iroot), int* ierflg);
};

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::math::tools;

// type definitions
typedef double value_type;  // or typedef float value_type;
typedef boost::numeric::ublas::vector<value_type> state_type;
typedef boost::numeric::ublas::matrix<value_type> matrix_type;
typedef rosenbrock4<value_type> stepper_type;

// constantes
const value_type pression = 0.1 / 760;  // atm soit 0.1 torr et 13 pascal
const value_type L = 3.e-2;             // distance entre deux plaques en m
const value_type pi = M_PI;
const value_type diff = pow((pi / L), 2.);   // facteur pour la diffusion
const value_type n_Ar = pression * 2.69e25;  // densite d'argon en m-3
const value_type n_SiH4_ini = n_Ar / 30.;    // densite de SiH4 initiale
const int Nbr_espece = 21;
const value_type DP =
    1.e23;  // eV/s.m3 puissance totale du systeme par unite de volume imposee
const float C = 1.35e21;  // m-3/s taux d'injection du SiH4 dans le réacteur
const value_type Tg = 0.02758;  // eV soit 320 K
const float D = 1.;  // m-3/s taux d'injection du SiH4 dans le réacteur

const int Nbr_K = 47;  // nombre d'equation dans le fichier
const int jmax = Nbr_K;
const int imax = 9;  // nombre de colonnes

struct Condition  // condition sur la bissection
{
  value_type tol = 1.e-9;
  bool operator()(value_type min, value_type max) {
    return abs(min - max) <= tol;
  }
};

struct ddriv_sys  // structure qui vient calculer les equations differentielles
{
  void operator()(int* neq, double* t, double* n, double* dndt) {
    /*0=e, 1=Armet, 2=SiH3-, 3=SiH2-, 4=SiH3+, 5=SiH4, 6=SiH3,
    7=H, 8=SiH2, 9=H2, 10=H2+, 11=Si2H5, 12=Si2H2, 13=Si2H4-,
    14=Si2H6, 15=Si2H3-, 16=Si2H5-, 17=SiH-, 18=SiH, 19=Si, 20=Arp, 21=NP*/

    value_type dSi = n[2] + n[3] + n[4] + n[5] + n[6] + n[8] + 2. * n[11] +
                     2. * n[12] + 2. * n[13] + 2. * n[14] + 2. * n[15] +
                     2. * n[16] + n[17] + n[18] +
                     n[19];  // somme de tous les atomes de si

    for (int k = 0; k < Nbr_espece; k++) {
      dndt[k] = 0;  // on initialise les equations a zero
    }

    for (int j = 0; j < jmax; j++) {
      p1 = static_cast<int>(Tab(0, j));  // perte 1 (reactif)
      p2 = static_cast<int>(Tab(1, j));  // perte 2 (reactif)
      g1 = static_cast<int>(Tab(2, j));  // gain 1 (produit)
      g2 = static_cast<int>(Tab(3, j));  // gain 2 (produit)
      g3 = static_cast<int>(Tab(4, j));  // gain 3 (produit)
      g4 = static_cast<int>(Tab(5, j));  // gain 4 (produit)

      Tp = (p2 == 0 or g3 == 0) ? Te : Tg;  // on defini si la temperature vaut
                                            // Tg ou Te en fonction de si les
                                            // electrons interviennent dans la
                                            // reaction
      Kt[j] = {Tab(6, j) * pow(Tp, Tab(7, j)) *
               exp(-Tab(8, j) / Tp)};  // coefficient de la loi d'arrhenius

      // on exprime les therme de gain ou pertes en fonction des especes qui
      // reagissent
      if (p1 == 200)  // un des deux reactifs est de l'argon (densite constante)
      {
        Tx = n_Ar * n[p2] * Kt[j];
      } else if (p2 == 100)  // il n'y a qu'un reactif
      {
        Tx = n[p1] * Kt[j];
      } else  // cas normal de deux reactifs qui peuvent etre egaux
      {
        Tx = n[p1] * n[p2] * Kt[j];
      }

      if (p1 != 200) {
        dndt[p1] = dndt[p1] - Tx;
      }
      if (p2 != 100) {
        dndt[p2] = dndt[p2] - Tx;
      }
      if (g1 != 200) {
        dndt[g1] = dndt[g1] + Tx;
      }
      if (g2 != 100) {
        dndt[g2] = dndt[g2] + Tx;
      }
      if (g3 != 100) {
        dndt[g3] = dndt[g3] + Tx;
      }
      if (g4 != 100) {
        dndt[g4] = dndt[g4] + Tx;
      }

    } /**/
      /*
      //introduction de la diffusion
      for (int a=0; a<Nbr_espece;a++)
      {if (a!=0 or a!=1 or a!=20) {dndt[a]=dndt[a]-C*n[a]/dSi;} //terme
      representant la pompe   if (a==0 or a==1 or a==4 or a==10 or a==20)
      {dndt[a]=dndt[a]-DA[a]*n[a];}} //diffusion
  

      dndt[5]=dndt[5]+C; //insertion de SiH4 dans le reacteur
      dndt[9]=dndt[9]+DA[10]*n[10]+DA[4]*n[4]; // H2+ + e -> H2 sur paroi //SiH3+
      + e -> SiH + H2 sur paroi   dndt[18]=dndt[18]+DA[4]*n[4]; //SiH3+ + e ->   SiH
      + H2sur paroi*/
  }
  /*state_type n;*/
  value_type Te;
  int p1, p2, g1, g2, g3, g4;
  value_type Tp, Tx, Tj;
  matrix_type Tab;
  state_type Kt;
};
// define a global for sys
ddriv_sys global_sys;

// calls the functor, static global
inline void ret_sys(int* n, double* t, double* yv, double* ydotv) {
  //   void operator()
  global_sys(n, t, yv, ydotv);
}

// dummy function, required by ddriv2_
inline void g(int* n, double* t, double* yv, int* iroot) {}

// calcul de la temperature a partir de la puissance dans le reacteur
struct etemperature {
  value_type operator()(value_type const& Te)

  {
    return -DP / n[0] + k(0, Te) * n_Ar * 16.14 + k(1, Te) * n_Ar * 12.31 +
           k(2, Te) * n[1] * 5.39 - k(3, Tg) * n[1] * n[1] * 8.48 -
           k(4, Te) * n[1] * 12.31 + k(5, Te) * n[5] * 10.68 +
           k(6, Te) * n[5] * 10.68 + k(7, Te) * n[5] * 8.29 +
           k(8, Te) * n[5] * 8.29 + k(9, Te) * n[5] * 24.1 +
           k(10, Te) * n[6] * 1.94 + k(11, Te) * n[6] * 1.30 +
           k(12, Te) * n[2] * 1.16 + k(13, Te) * n[3] * 1.16 +
           k(14, Te) * n[8] * 1.5 * Te + k(15, Te) * n[9] * 10.09 +
           k(16, Te) * n[9] * 16.05 + k(42, Te) * n[6] * 1.5 * Te +
           k(43, Te) * n[18] * 1.5 * Te + k(44, Te) * n[17] * 1.25;
  }

  state_type n;

  // fonction pour calculer les K en faisant
  // varier Te dans la bissection
  value_type k(int ind,
               value_type Tp) {
    value_type K;
    K = Tab(6, ind) * pow(Tp, Tab(7, ind)) * exp(-Tab(8, ind) / Tp);
    return K;
  }

  // Tab as member
  matrix_type Tab;
};

// ecriture des densite dans un fichier
void write_density(const value_type t, const value_type Te,
                   const state_type& n) {
  cout << t << '\t' << Te << '\t' << n[0] << '\t' << n[1] << '\t' << n[2]
       << '\t' << n[3] << '\t' << n[4] << '\t' << n[5] << '\t' << n[6] << '\t'
       << n[7] << '\t' << n[8] << '\t' << n[9] << '\t' << n[10] << '\t' << n[11]
       << '\t' << n[12] << '\t' << n[13] << '\t' << n[14] << '\t' << n[15]
       << '\t' << n[16] << '\t' << n[17] << '\t' << n[18] << '\t' << n[19]
       << '\t' << n[20] << '\t' << n[13] + n[16] << endl;
}

void write_density(ofstream& fp, const value_type t, const value_type Te,
                   const state_type &n) {
  fp << t << '\t' << Te << '\t' << n[0] << '\t' << n[1] << '\t' << n[2]
       << '\t' << n[3] << '\t' << n[4] << '\t' << n[5] << '\t' << n[6] << '\t'
       << n[7] << '\t' << n[8] << '\t' << n[9] << '\t' << n[10] << '\t' << n[11]
       << '\t' << n[12] << '\t' << n[13] << '\t' << n[14] << '\t' << n[15]
       << '\t' << n[16] << '\t' << n[17] << '\t' << n[18] << '\t' << n[19]
       << '\t' << n[20] << '\t' << n[13] + n[16] << endl;
}

int main(int argc, char** argv) {
  // output stream
  ofstream outfile;
  outfile.open("dens.dat");

  // lecture dans le fichier contenant les reactions et les coefficients pour
  // arrhenius
  ifstream fichier_k("fichagarwal.dat");

  matrix_type Tab(imax, jmax);

  if (fichier_k) {
    // Tout est prêt pour la lecture.
    cerr << "fichier ouvert" << endl;

    for (int j = 0; j < jmax; j++) {
      fichier_k >> Tab(0, j) >> Tab(1, j) >> Tab(2, j) >> Tab(3, j) >>
          Tab(4, j) >> Tab(5, j) >> Tab(6, j) >> Tab(7, j) >> Tab(8, j);
    }
    fichier_k.close();

  } else {
    cerr << "ERREUR: Impossible d'ouvrir le fichier en lecture." << endl;
    return -1;
  }

  value_type Te = 0.7;  // valeur initiale de la temperature

  // legende
  cout << "t" << '\t' << "Te" << '\t' << "e" << '\t' << "Armet" << '\t'
       << "SiH3m" << '\t' << "SiH2" << '\t' << "SiH3p" << '\t' << "SiH4" << '\t'
       << "SiH3" << '\t' << "H" << '\t' << "SiH2" << '\t' << "H2" << '\t'
       << "H2p" << '\t' << "Si2H5" << '\t' << "Si2H2" << '\t' << "Si2H4m"
       << '\t' << "Si2H6" << '\t' << "Si2H3m" << '\t' << "Si2H5m" << '\t'
       << "SiHm" << '\t' << "SiH" << '\t' << "Si" << '\t' << "Arp" << '\t'
       << "NP" << endl;

  // vecteur de densite et conditions initiales
  state_type n_ini(Nbr_espece, 0.0);
  n_ini[0] = 1.e16;
  n_ini[1] = 1.e10;
  n_ini[5] = n_SiH4_ini;
  n_ini[20] = 1.e16;
  n_ini[4] = n_ini[0] - n_ini[20];

  state_type DL(Nbr_espece, 0.0);  // vecteur de diffusion libre en m2/s
  state_type mu(Nbr_espece, 0.0);  // vecteur de mobilite en m2/(V.s)
  state_type DA(Nbr_espece, 0.0);  // vecteur de diffusion ambipolaire en m2/s

  state_type Kt(jmax, 0.0);

  // Coefficients de diffusion libres de Chapman-Enskog
  value_type D_mol = 2.;  // diametre de (molecule + argon)/2 en A
  value_type D_e = 1.;    // diametre de (electron+ argon)/2 en A

  value_type CE_e = 1.858e-3 * (Tg * 1.1604e4) * sqrt(Te * 1.1604e4) /
                    (pression * pow(D_e, 2.)) *
                    1.e-4;  // on met les temperatures en K et on convertis pour
                            // l'avoir en m2/s (*1.e-4)
  value_type CE_mol = 1.858e-3 * pow((Tg * 1.1604e4), 3. / 2.) /
                      (pression * pow(D_mol, 2.)) *
                      1.e-4;  // on met les temperatures en K et on convertis
                              // pour l'avoir en m2/s (*1.e-4)

  /*DL[0]= CE_e*sqrt(1836.2 + 1./40.)/2.; // OMEGA = 2 et masse atomique de
  l'electron =1/1836.2 mu[0]= DL[0]/Te; //Te en eV et DL en m2/s */
  DL[0] = 120. / (pression * 760.);  // valeur de benjamin
  mu[0] = DL[0] / Te;

  DL[1] = 0.075;  // valeur de benjamin en m2/s

  DL[4] = CE_mol * sqrt(1. / 31. + 1. / 40.) / 10.;  // OMEGA = 10
  mu[4] = DL[4] / Tg;

  DL[10] = CE_mol * sqrt(1. / 2. + 1. / 40.) / 10.;  // OMEGA = 10
  mu[10] = DL[10] / Tg;

  /*DL[20]= CE_mol*sqrt(2./40.)/10.; // OMEGA = 10
  mu[20]= DL[20]/Tg;*/
  DL[20] = 4.e-3 / (pression * 760.);  // valeur de benjamin
  mu[20] = DL[20] / Tg;

  // variable du temps
  double t = 0.0;
  value_type dt = 1.0e-8;
  value_type Tmax = 20.e-3;
  value_type NT = Tmax / dt;

  // variable pour la bissection
  value_type min = Tg;
  value_type max = 10.0;
  boost::uintmax_t max_iter = 500;
  eps_tolerance<value_type> tol(30);

  state_type n_new(Nbr_espece, 0.0);  // initialisation du vecteur densite
  n_new = n_ini;
  state_type n_err(Nbr_espece, 0.0);  // error

  // declare la fonction etemperature
  etemperature etemp;

  // assigne les valeur a la fonction etemp
  etemp.n = n_ini;

  // set Tab in etemp
  etemp.Tab = Tab;

  // premier calcul de Te
  pair<value_type, value_type> pair_Te =
      toms748_solve(etemp, min, max, tol, max_iter);

  Te = pair_Te.first;
  cerr << "\n[ii]  Temperature Initiale = " << Te << endl;

  // global_sys.n=n_ini;
  global_sys.Tab = Tab;
  global_sys.Kt = Kt;
  /*
  //verification du coefficient d'arrhenius et de la temperature
  for (int j=0;j<jmax;j++)
  {
   p2=Tab[1][j];
   g3=Tab[4][j];
   Tp=(p2==0 or g3==0)?Te:Tg;
   Kt[j]={Tab[6][j]*pow(Tp,Tab[7][j])*exp(-Tab[8][j]/Tp)};
  cerr<<j<<'\t'<<Tp<<'\t'<<Kt[j]<<endl;
  }

  cerr<<k(44,Te)<<endl;
  cerr<<k45(Te)<<endl;
  */

  //   ddriv_sys dsys;
  int n = Nbr_espece;
  int mstate = -1;
  int nroot = 0;
  double eps = 1.e-6;
  double ewt = 1.e-10;
  int mint = 3;

  int lenw = n * n + 17 * n + 252;
  std::vector<double> work;

  int leniw = n + 50;
  std::vector<double> iwork;

  work.resize(lenw);
  iwork.resize(leniw);

  int ierflg;

  for (int i = 1; i <= NT + 1; i++) {
    // calcul des coefficients pour la diffusion ambipolaire
    value_type n_mu = n_new[0] * mu[0] + n_new[4] * mu[4] + n_new[10] * mu[10] +
                      n_new[20] * mu[20];
    value_type n_DL = -n_new[0] * DL[0] + n_new[4] * DL[4] +
                      n_new[10] * DL[10] + n_new[20] * DL[20];

    // diffusion ambipolaire
    DA[0] = (DL[0] + mu[0] * n_DL / n_mu) * diff;  // s-1
    DA[1] = DL[1] * diff;
    DA[4] = (DL[4] - mu[4] * n_DL / n_mu) * diff;
    DA[10] = (DL[10] - mu[10] * n_DL / n_mu) * diff;
    DA[20] = (DL[20] - mu[20] * n_DL / n_mu) * diff;

    global_sys.Te = Te;

    double tf = t + dt;

    ddriv2_(&n, &t, &n_new[0], ret_sys, &tf, &mstate, &nroot, &eps, &ewt, &mint,
            &work[0], &lenw, &iwork[0], &leniw, g, &ierflg);
//     cerr << "patate1" << endl;
    // assignation des valeur a la fonction etemp
    etemp.n = n_new;
    if (i % ((int)(NT / 100)) == 0) {
      write_density(outfile, t, Te, n_new);
    }
    // trouver un noyuveau Te
    pair<value_type, value_type> pair_Te =
        toms748_solve(etemp, min, max, tol, max_iter);

    Te = pair_Te.first;
//     t += dt;
    n_ini = n_new;  // update
  }

  value_type charge =
      (n_new[20] + n_new[4] + n_new[10] - n_new[0] - n_new[2] - n_new[3] -
       n_new[13] - n_new[15] - n_new[16] - n_new[17]) /
      (n_new[20] + n_new[4] + n_new[10]);

  cerr << "charge/dArp=" << charge << endl;

  value_type Si =
      (n_new[2] + n_new[3] + n_new[4] + n_new[13] * 2 + 2 * n_new[15] +
       n_new[16] * 2 + n_new[17] + n_new[5] + n_new[6] + n_new[8] + n_new[18] +
       2 * n_new[11] + n_new[19] + n_new[12] * 2 + n_new[14] * 2) /
      n_SiH4_ini;

  cerr << "Si=" << Si << endl;

  value_type H =
      (3 * n_new[2] + 2 * n_new[3] + 3 * n_new[4] + 2 * n_new[10] +
       4 * n_new[13] + 3 * n_new[15] + 5 * n_new[16] + n_new[17] +
       4 * n_new[5] + 3 * n_new[6] + n_new[7] + 2 * n_new[8] + 2 * n_new[9] +
       n_new[18] + 5 * n_new[11] + 2 * n_new[12] + 6 * n_new[14]) /
      (4 * n_SiH4_ini);

  cerr << "H=" << H << endl;

  return 0;
}
