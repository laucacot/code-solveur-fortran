
#include <cmath>
#include <iostream>
#include <string>
//#include <boost/serialization/array_wrapper.hpp>

#include <vector>
// Optimization
#define NDEBUG
#define BOOST_UBLAS_NDEBUG
#include <stdlib.h>
#include <time.h>
#include <boost/array.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <fstream>

extern "C" {
void ddriv2_(int* n, double* tv, double* yv,
             void(f)(int* n, double* t, double* yv, double* ydotv),
             double* tout, int* mstate, int* nroot, double* eps, double* ewt,
             int* mint, double* work, int* lenw, double* iwork, int* leniw,
             void(g)(int* n, double* t, double* yv, int* iroot), int* ierflg);
};

using namespace std;
using namespace boost::math::tools;

// type definitions
using namespace boost::numeric;
typedef boost::numeric::ublas::vector<double> state_type;
typedef boost::numeric::ublas::matrix<double> matrix_type;

// constantes
const double pression = 0.1 / 760;  // atm soit 0.1 torr et 13 pascal
const double L = 3.e-2;             // distance entre deux plaques en m
const double pi = M_PI;
const double diff = pow((pi / L), 2.) * 2.;  // facteur pour la diffusion
const double n_Ar = pression * 2.69e25;      // densite d'argon en m-3
const double n_SiH4_ini = n_Ar / 30.;        // densite de SiH4 initiale
const int Nbr_espece = 24;
// const double DP =1.e23;  // eV/s.m3 puissance totale du systeme par unite de
// volume imposee
const double C = 1.35e21;   // m-3/s taux d'injection du SiH4 dans le réacteur
const double Tg = 0.02758;  // eV soit 320 K

const int Nbr_K = 47;  // nombre d'equation dans le fichier
const int jmax = Nbr_K;
const int imax = 9;  // nombre de colonnes

const double e0 = 1.6022e-19;    // coulomb
const double eps0 = 8.8542e-12;  // s2.C2/kg/m3
const double cVfl = e0 / (4 * pi * eps0);
const int V = 30.;  // V voltage applique

const double rNP = 1.e-9;  // m rayon initial des NP
const double qNP = 0.;     // C charge initiale des NP
const double nNP = 0.;     // 2.e15; //m-3 densite des NP

// volume d'un atome de Si pour densite solide 2.33 g
const double vSi = (4. / 3.) * pi * pow(1.36e-10, 3);
const double rate = C / n_SiH4_ini;  // taux d'injection et d'evacuation

struct Condition  // condition sur la bissection
{
  double tol = 1.e-9;
  bool operator()(double min, double max) { return abs(min - max) <= tol; }
};

struct ddriv_sys  // structure qui vient calculer les equations differentielles
{
  void operator()(int* neq, double* t, double* n, double* dndt) {
    /*0=e, 1=Armet, 2=SiH3-, 3=SiH2-, 4=SiH3+, 5=SiH4, 6=SiH3,
    7=H, 8=SiH2, 9=H2, 10=H2+, 11=Si2H5, 12=Si2H2, 13=Si2H4-,
    14=Si2H6, 15=Si2H3-, 16=Si2H5-, 17=SiH-, 18=SiH, 19=Si, 20=Arp, 21=NP*/
    double dpp = n[4] + n[10] + n[20];
    double dnn = n[0] + n[2] + n[3] + n[13] + n[15] + n[16] + n[17];
    double ratep =
        rate * dnn / dpp;  // pour que les charges sortent par paires = et -
    double Vfl =
        cVfl * n[22] / n[21];  // V potentiel flottant e*qNP / 4 pi eps0 rNP
    double sNP = pi * pow(n[21], 2);  // surface de la NP
    state_type DA(Nbr_espece, 0.0);  // vecteur de diffusion ambipolaire en m2/s
                                     // if (Vfl<0.){
    double facte = sNP * exp(Vfl / Te);  // qNP<0
    double factn = sNP * exp(Vfl / Tg);
    double factp = sNP * (1. - Vfl / Tg); /*}
else { facte=sNP*(1.+Vfl/Te); //else qNP>0
            factn=sNP*(1.+Vfl/Tg);
            factp=sNP*exp(-Vfl/Tg);}*/
    // calcul des coefficients pour la diffusion ambipolaire
    double n_mu = n[0] * mu[0] + n[4] * mu[4] + n[10] * mu[10] + n[20] * mu[20];
    double n_DL =
        -n[0] * DL[0] + n[4] * DL[4] + n[10] * DL[10] + n[20] * DL[20];

    // diffusion ambipolaire
    DA[0] = (DL[0] + mu[0] * n_DL / n_mu);  // s-1
    DA[1] = DL[1];
    DA[4] = (DL[4] - mu[4] * n_DL / n_mu);
    DA[10] = (DL[10] - mu[10] * n_DL / n_mu);
    DA[20] = (DL[20] - mu[20] * n_DL / n_mu);

    // chargement des NP
    double ffe = facte * vth[0] * n[0];
    double ffn2 = factn * vth[2] * n[2];
    double ffn3 = factn * vth[3] * n[3];
    double ffp4 = factp * vth[4] * n[4];
    double ffp10 = factp * vth[10] * n[10];
    double ffn13 = factn * vth[13] * n[13];
    double ffn15 = factn * vth[15] * n[15];
    double ffn16 = factn * vth[16] * n[16];
    double ffn17 = factn * vth[17] * n[17];
    double ffp20 = factp * vth[20] * n[20];

    double ccol = nNP * 4. * sNP / vSi;  // 4*sNP=surface de la NP

    /*double dSi = n[2] + n[3] + n[4] + n[5] + n[6] + n[8] + 2. * n[11] +
                     2. * n[12] + 2. * n[13] + 2. * n[14] + 2. * n[15] +
                     2. * n[16] + n[17] + n[18] +
                     n[19];  // somme de tous les atomes de si*/

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
    + e -> SiH + H2 sur paroi   dndt[18]=dndt[18]+DA[4]*n[4]; //SiH3+ + e -> SiH
    + H2sur paroi*/
    // introduction de la diffusion
    for (int a = 0; a < Nbr_espece; a++) {
      if (a == 0 or a == 1 or a == 4 or a == 10 or a == 20) {
        dndt[a] = dndt[a] - DA[a] * n[a];
      }
      if (a != 4 or a != 10 or a < 20) {
        dndt[a] = dndt[a] - rate * n[a];
      }  // pompe sur les especes non +
      // if (a==4 or a==10 or a==20){dndt[a]=dndt[a]-ratep*n[a];}//pompe sur les
      // especes +
    }  // diffusion

    /*  dndt[5] = dndt[5] + C;  // insertion de SiH4 dans le reacteur
       dndt[9] = dndt[9] + DA[10] * n[10] + DA[4] * n[4];
       // H2+ + e -> H2 sur paroi
       //SiH3+ + e -> SiH + H2 sur paroi
       dndt[18] = dndt[18] + DA[4] * n[4];  // SiH3+ + e ->  SiH + H2sur paroi*/

    dndt[0] = dndt[0] - ffe * nNP;
    dndt[1] = dndt[1] - sNP * vth[1] * nNP * n[1];
    dndt[2] = dndt[2] - ffn2 * nNP;
    dndt[3] = dndt[3] - ffn3 * nNP;
    dndt[4] = dndt[4] - ffp4 * nNP;
    dndt[5] = dndt[5] - ccol * coll[5] * n[5] +
              C;  // insertion de SiH4 dans le reacteur
    dndt[6] = dndt[6] - ccol * coll[6] * n[6];
    //+DA_[4]*n[4];
    dndt[7] = dndt[7] + (ffn2 + ffp4 + ffn15 + ffn16 + ffn17) * nNP +
              ccol * (coll[6] * n[6] + coll[11] * n[11] + coll[18] * n[18]);
    //+0.3*DA_[4]*n[4];//SiH3+ + e -> Si + H + H2 sur paroi
    dndt[8] = dndt[8] - ccol * coll[8] * n[8];
    //+DA_[4]*n[4];//SiH3+ + e -> SiH2 + H sur paroi
    dndt[9] =
        dndt[9] + DA[10] * n[10]  // H2+ + e -> H2 sur paroi
        + DA[4] * n[4]  // SiH3+ + e -> SiH + H2, Si + H + H2 sur paroi
        + (ffn2 + ffn3 + ffp4 + ffp10 + 2. * ffn13 + ffn15 + 2. * ffn16) * nNP +
        ccol *
            (2. * coll[5] * n[5] + coll[6] * n[6] + coll[8] * n[8] +
             2. * coll[11] * n[11] + coll[12] * n[12] + 3. * coll[14] * n[14]);
    dndt[10] = dndt[10] - ffp10 * nNP;
    dndt[11] = dndt[11] - ccol * coll[11] * n[11];
    dndt[12] = dndt[12] - ccol * coll[12] * n[12];
    dndt[13] = dndt[13] - ffn13 * nNP;
    dndt[14] = dndt[14] - ccol * coll[14] * n[14];
    dndt[15] = dndt[15] - ffn15 * nNP;
    dndt[16] = dndt[16] - ffn16 * nNP;
    dndt[17] = dndt[17] - ffn17 * nNP;
    dndt[18] = dndt[18] + DA[4] * n[4]  //SiH3+ + e ->  SiH + H2 sur paroi
               - ccol * coll[18] * n[18];
    dndt[19] = dndt[19] - ccol * coll[19] * n[19];
    //+0.3*DA_[4]*n[4];//SiH3+ + e ->  Si + H + H2 sur paroi
    dndt[20] = dndt[20] - ffp20 * nNP;

    // Collage dR/dt sur les NP:
    dndt[21] =
        dndt[21] + coll[5] * n[5] + coll[6] * n[6] + coll[8] * n[8] +
        2. * coll[11] * n[11] + 2. * coll[12] * n[12] + 2. * coll[14] * n[14] +
        coll[18] * n[18] + coll[19] * n[19] +
        (ffn2 + ffn3 + 2. * ffn13 + 2. * ffn15 + 2. * ffn16 + ffn17 + ffp4) *
            vSi / sNP;

    // Charge des NP par OML:
    dndt[22] = dndt[22] - ffe - ffn2 - ffn3 - ffn13 - ffn15 - ffn16 - ffn17 +
               ffp4 + ffp10 + ffp20;

    // Densite des NP:
    dndt[23] = dndt[23] + dndt[13] + dndt[16];
  }
  state_type n;
  double Te;
  double DP;
  int p1, p2, g1, g2, g3, g4;
  double Tp, Tx, Tj;
  matrix_type Tab;
  state_type Kt;
  state_type DL;
  state_type mu;
  state_type coll;
  state_type vth;
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
  double operator()(double const& Te)

  {  // calcul des coefficients pour la diffusion ambipolaire
    double n_mu = mu[0] * n[0] + n[4] * mu[4] + n[10] * mu[10] + n[20] * mu[20];
    double n_DL =
        -n[0] * DL[0] + n[4] * DL[4] + n[10] * DL[10] + n[20] * DL[20];

    // diffusion ambipolaire
    double Diffe = (DL[0] + mu[0] * n_DL / n_mu);  // s-1

    double Vfl =
        cVfl * n[22] / n[21];  // V potentiel flottant e*qNP / 4 pi eps0 rNP
    double sNP = pi * pow(n[21], 2);  // surface de la NP
    // if (Vfl<=0.){
    double Fe = sNP * vth[0] * exp(Vfl / Te);
    /*}
else { Fe=sNP*vth[0]*(1.-Vfl/Te);}*/

    return -DP / Te + k(0, Te) * n_Ar * 15.76 + k(1, Te) * n_Ar * 11.76 +
           k(2, Te) * n[1] * 4. - k(4, Te) * n[1] * 11.76 +
           k(5, Te) * n[5] * 10.68 + k(6, Te) * n[5] * 10.68 +
           k(7, Te) * n[5] * 8.29 + k(8, Te) * n[5] * 8.29 +
           k(9, Te) * n[5] * 12 + k(10, Te) * n[6] * 1.94 +
           k(11, Te) * n[6] * 1.30 + k(12, Te) * n[2] * 1.16 +
           k(13, Te) * n[3] * 1.16 + k(14, Te) * n[8] * 1.5 * Te +
           k(15, Te) * n[9] * 10.09 + k(16, Te) * n[9] * 16.05 +
           k(42, Te) * n[6] * 1.5 * Te + k(43, Te) * n[18] * 1.5 * Te +
           k(44, Te) * n[17] * 1.25 + Diffe * 1.5 * Te /*perte sur les parois*/
           + Fe * nNP * 1.5 * Te                       /*pertes sur les NP*/
           + rate * 1.5 * Te; /*perte taux injection evacuation*/

    /*
        return -DP / n[0] + k(0, Te) * n_Ar * 16.14 + k(1, Te) * n_Ar * 12.31 +
               k(2, Te) * n[1] * 5.39 - k(3, Tg) * n[1] * n[1] * 8.48 -
               k(4, Te) * n[1] * 12.31 + k(5, Te) * n[5] * 10.68 +
               k(6, Te) * n[5] * 10.68 + k(7, Te) * n[5] * 8.29 +
               k(8, Te) * n[5] * 8.29 + k(9, Te) * n[5] * 24.1 +
               k(10, Te) * n[6] * 1.94 + k(11, Te) * n[6] * 1.30 +
               k(12, Te) * n[2] * 1.16 + k(13, Te) * n[3] * 1.16 +
               k(14, Te) * n[8] * 1.5 * Te + k(15, Te) * n[9] * 10.09 +
               k(16, Te) * n[9] * 16.05 + k(42, Te) * n[6] * 1.5 * Te +
               k(43, Te) * n[18] * 1.5 * Te + k(44, Te) * n[17] * 1.25;*/
  }

  state_type n;
  state_type DL;
  state_type mu;
  state_type vth;
  double DP;
  // fonction pour calculer les K en faisant
  // varier Te dans la bissection
  double k(int ind, double Tp) {
    double K;
    K = Tab(6, ind) * pow(Tp, Tab(7, ind)) * exp(-Tab(8, ind) / Tp);
    return K;
  }

  // Tab as member
  matrix_type Tab;
};

// ecriture des densite dans un fichier

void write_density(ofstream& fp, const double& t, const double& Te,
                   const state_type& n) {
  fp << t << '\t' << Te << '\t' << n[0] << '\t' << n[1] << '\t' << n[2] << '\t'
     << n[3] << '\t' << n[4] << '\t' << n[5] << '\t' << n[6] << '\t' << n[7]
     << '\t' << n[8] << '\t' << n[9] << '\t' << n[10] << '\t' << n[11] << '\t'
     << n[12] << '\t' << n[13] << '\t' << n[14] << '\t' << n[15] << '\t'
     << n[16] << '\t' << n[17] << '\t' << n[18] << '\t' << n[19] << '\t'
     << n[20] << '\t' << n[13] + n[16] << endl;
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

  double Te = 3.;  // valeur initiale de la temperature

  // legende
  outfile << "#t" << '\t' << "Te" << '\t' << "e" << '\t' << "Armet" << '\t'
          << "SiH3m" << '\t' << "SiH2" << '\t' << "SiH3p" << '\t' << "SiH4"
          << '\t' << "SiH3" << '\t' << "H" << '\t' << "SiH2" << '\t' << "H2"
          << '\t' << "H2p" << '\t' << "Si2H5" << '\t' << "Si2H2" << '\t'
          << "Si2H4m" << '\t' << "Si2H6" << '\t' << "Si2H3m" << '\t' << "Si2H5m"
          << '\t' << "SiHm" << '\t' << "SiH" << '\t' << "Si" << '\t' << "Arp"
          << '\t' << "NP" << endl;

  // vecteur de densite et conditions initiales
  state_type n_ini(Nbr_espece, 0.0);
  n_ini[0] = 1.e16;
  n_ini[1] = 1.e16;
  n_ini[5] = n_SiH4_ini;
  n_ini[20] = n_ini[0] - n_ini[4];
  n_ini[21] = rNP;
  n_ini[22] = qNP;

  // determination du collage sur les NP
  state_type coll(Nbr_espece, 0.0);  // vecteur collage
  state_type vth(Nbr_espece, 0.0);   // vecteur vitesse thermique

  double cvth = 1.56e4 * sqrt(Tg);  // NRL * sqrt (8/pi)

  vth[0] = 6.69e5 * sqrt(Te);
  vth[1] = cvth / sqrt(40.);
  vth[2] = cvth / sqrt(31.);
  vth[3] = cvth / sqrt(30.);
  vth[4] = cvth / sqrt(33.);
  vth[5] = cvth / sqrt(34.);
  vth[6] = cvth / sqrt(33.);
  vth[7] = cvth / sqrt(1.);
  vth[8] = cvth / sqrt(30.);
  vth[9] = cvth / sqrt(2.);
  vth[10] = cvth / sqrt(2.);
  vth[11] = cvth / sqrt(61.);
  vth[12] = cvth / sqrt(58.);
  vth[13] = cvth / sqrt(60.);
  vth[14] = cvth / sqrt(62.);
  vth[15] = cvth / sqrt(59.);
  vth[16] = cvth / sqrt(61.);
  vth[17] = cvth / sqrt(29.);
  vth[18] = cvth / sqrt(29.);
  vth[19] = cvth / sqrt(28.);
  vth[20] = cvth / sqrt(40.);

  coll[5] = 1.e-5 * vSi * vth[5];  // m4/s  coefficients de collage de Le Picard
  coll[6] = 0.045 * vSi * vth[6];
  coll[8] = 0.8 * vSi * vth[8];
  coll[11] = 1.e-5 * vSi * vth[11];  // pas dans Le Picard
  coll[12] = 1.e-5 * vSi * vth[12];  // pas dans Le Picard
  coll[14] = 1.e-5 * vSi * vth[14];
  coll[18] = 0.95 * vSi * vth[18];
  coll[19] = 1.0 * vSi * vth[19];  // pas dans Le Picard

  state_type DL(Nbr_espece, 0.0);  // vecteur de diffusion libre en m2/s
  state_type mu(Nbr_espece, 0.0);  // vecteur de mobilite en m2/(V.s)

  state_type Kt(jmax, 0.0);

  // Coefficients de diffusion libres de Chapman-Enskog
  double D_mol = 2.;  // diametre de (molecule + argon)/2 en A
  double D_e = 1.;    // diametre de (electron+ argon)/2 en A

  double CE_e = 1.858e-3 * (Tg * 1.1604e4) * sqrt(Te * 1.1604e4) /
                (pression * pow(D_e, 2.)) *
                1.e-4;  // on met les temperatures en K et on convertis pour
                        // l'avoir en m2/s (*1.e-4)
  double CE_mol = 1.858e-3 * pow((Tg * 1.1604e4), 3. / 2.) /
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

  double DP = 0.5 * DL[0] * pow((V / L), 2.);

  for (int i = 0; i < Nbr_espece; i++) {
    DL[i] = DL[i] * diff;
    mu[i] = mu[i] * diff;
  }

  // variable du temps
  double t = 0.0;
  double dt = 10.0e-10;
  double Tmax = 2e-7;//20.e-3;
  double NT = Tmax / dt;

  // variable pour la bissection
  double min = Tg;
  double max = 1000.;
  boost::uintmax_t max_iter = 1000;
  eps_tolerance<double> tol(30);

  state_type n_new(Nbr_espece, 0.0);  // initialisation du vecteur densite
  n_new = n_ini;
  state_type n_err(Nbr_espece, 0.0);  // error

  // declare la fonction etemperature
  etemperature etemp;

  // assigne les valeur a la fonction etemp
  etemp.n = n_ini;

  // set Tab in etemp
  etemp.Tab = Tab;

  // assigne les vecteurs Dl  mu et vth a la fonction etemp
  etemp.DL = DL;
  etemp.mu = mu;
  etemp.vth = vth;

  // assigne DP a etemp
  etemp.DP = DP;

  // premier calcul de Te
  pair<double, double> pair_Te = toms748_solve(etemp, min, max, tol, max_iter);

  Te = pair_Te.first;
  cerr << "\n[ii]  Temperature Initiale = " << Te << endl;

  // global_sys.n=n_ini;
  global_sys.Tab = Tab;
  global_sys.Kt = Kt;

  // assignation a global sys de DP vth, mu, DL et coll
  global_sys.n = n_ini;
  global_sys.DP = DP;
  global_sys.vth = vth;
  global_sys.mu = mu;
  global_sys.DL = DL;
  global_sys.coll = coll;
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

  for (int i = 0; i <= NT + 1; i++) {
    mu[1] = DL[1] / Te;           // 30./(pression*760.)     m2/(V.s)
    vth[1] = 6.69e-5 * sqrt(Te);  // m/s  vitesse thermique electrons
 
    global_sys.Te = Te;

    double tf = t + dt;

    ddriv2_(&n, &t, &n_new[0], ret_sys, &tf, &mstate, &nroot, &eps, &ewt, &mint,
            &work[0], &lenw, &iwork[0], &leniw, g, &ierflg);
//     //     cerr << "patate1" << endl;
//     // assignation des valeur a la fonction etemp
    etemp.n = n_new;
    if (i % ((int)(NT / 100)) == 0) {
      write_density(outfile, tf, Te, n_new);
    }

    // trouver un noyuveau Te
    pair<double, double> pair_Te =
        toms748_solve(etemp, min, max, tol, max_iter);

    Te = pair_Te.first;
    //     t += dt;
    n_ini = n_new;  // update
  }

  double charge = (n_new[20] + n_new[4] + n_new[10] - n_new[0] - n_new[2] -
                   n_new[3] - n_new[13] - n_new[15] - n_new[16] - n_new[17]) /
                  (n_new[20] + n_new[4] + n_new[10]);

  cerr << "charge/dArp=" << charge << endl;

  double Si =
      (n_new[2] + n_new[3] + n_new[4] + n_new[13] * 2 + 2 * n_new[15] +
       n_new[16] * 2 + n_new[17] + n_new[5] + n_new[6] + n_new[8] + n_new[18] +
       2 * n_new[11] + n_new[19] + n_new[12] * 2 + n_new[14] * 2) /
      n_SiH4_ini;

  cerr << "Si=" << Si << endl;

  double H =
      (3 * n_new[2] + 2 * n_new[3] + 3 * n_new[4] + 2 * n_new[10] +
       4 * n_new[13] + 3 * n_new[15] + 5 * n_new[16] + n_new[17] +
       4 * n_new[5] + 3 * n_new[6] + n_new[7] + 2 * n_new[8] + 2 * n_new[9] +
       n_new[18] + 5 * n_new[11] + 2 * n_new[12] + 6 * n_new[14]) /
      (4 * n_SiH4_ini);

  cerr << "H=" << H << endl;

  return 0;
}
