
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
const value_type diff = pow((pi / L), 2.) * 2.;  // facteur pour la diffusion
const value_type n_Ar = pression * 2.69e25;      // densite d'argon en m-3
const value_type n_SiH4_ini = n_Ar / 30.;        // densite de SiH4 initiale
const int Nbr_espece = 24;
// const value_type DP =1.e23;  // eV/s.m3 puissance totale du systeme par unite
// de volume imposee
const float C = 2.e21;  // m-3/s taux d'injection du SiH4 dans le réacteur
const value_type Tg = 0.02758;  // eV soit 320 K

const int Nbr_K = 47;  // nombre d'equation dans le fichier
const int jmax = Nbr_K;
const int imax = 9;  // nombre de colonnes

const value_type e0 = 1.6022e-19;    // coulomb
const value_type eps0 = 8.8542e-12;  // s2.C2/kg/m3
const value_type cVfl = e0 / (4 * pi * eps0);
const int V = 30.;  // V voltage applique

const value_type rNP = 1.e-9;  // m rayon initial des NP
const value_type qNP = 0.;     // C charge initiale des NP
const value_type nNP = 0.;     // 2.e15; //m-3 densite des NP

const value_type vSi =
    (4. / 3.) * pi *
    pow(1.36e-10, 3);  // volume d'un atome de Si pour densite solide 2.33 g
const value_type rate = C / n_SiH4_ini;  // taux d'injection et d'evacuation

struct Condition  // condition sur la bissection
{
  value_type tol = 1.e-9;
  bool operator()(value_type min, value_type max) {
    return abs(min - max) <= tol;
  }
};

// calcul des K dependant de Te

value_type k1(value_type Te)  // K1 Ar + e -> Ar+ + 2e
{
  value_type K1;
  K1 = 7.06E-17 * pow((Te), 0.6) * exp(-(16.14) / (Te));
  return K1;
}

value_type k2(value_type Te)  // K2 Ar + e -> Ar* + e
{
  value_type K2;
  K2 = 11.69E-15 * exp(-(12.31) / (Te));
  return K2;
}

value_type k3(value_type Te)  // K3 Ar* + e -> Ar+ + 2e
{
  value_type K3;
  K3 = 124.92E-15 * exp(-(5.39) / (Te));
  return K3;
}

value_type k4(value_type Te)  // K4 Ar* + Ar* -> Ar + Ar+ + e
{
  value_type K4;
  K4 = 6.144e-16;
  return K4;
}

value_type k5(value_type Te)  // K5 Ar* + e -> Ar + e
{
  value_type K5;
  K5 = 431.89E-18 * pow((Te), 0.74);
  return K5;
}

value_type k6(value_type Te)  // K6 SiH4 + e -> SiH3 + H + e
{
  value_type K6;
  K6 = 1.83E-9 * pow((Te), -1) * exp(-(10.68) / (Te));
  return K6;
}

value_type k7(value_type Te)  // K7 SiH4 + e -> SiH2 + 2H + e
{
  value_type K7;
  K7 = 8.97E-9 * pow((Te), -1) * exp(-(10.68) / (Te));
  return K7;
}

value_type k8(value_type Te)  // K8 SiH4 + e -> SiH3- + H
{
  value_type K8;
  K8 = 3.77E-9 * pow((Te), -1.63) * exp(-(8.29) / (Te));
  return K8;
}

value_type k9(value_type Te)  // K9 SiH4 + e -> SiH2- + 2H
{
  value_type K9;
  K9 = 3.77E-9 * pow((Te), -1.63) * exp(-(8.29) / (Te));
  return K9;
}

value_type k10(value_type Te)  // K10 SiH4 + e -> SiH3+ + H + 2e
{
  value_type K10;
  K10 = 2.50E2 * pow((Te), -2.93) * exp(-(24.1) / (Te));
  return K10;
}

value_type k11(value_type Te)  // K11 SiH3 + e -> SiH2- + H
{
  value_type K11;
  K11 = 5.71E-9 * pow((Te), -0.5) * exp(-(1.94) / (Te));
  return K11;
}

value_type k12(value_type Te)  // K12 SiH3 + e -> SiH3+  + 2e
{
  value_type K12;
  K12 = 2.26E-16 * pow((Te), 0.5) * exp(-(1.30) / (Te));
  return K12;
}

value_type k13(value_type Te)  // K13 SiH3- + e -> SiH3 + 2e
{
  value_type K13;
  K13 = 3.15E-16 * pow((Te), 0.5) * exp(-(1.16) / (Te));
  return K13;
}

value_type k14(value_type Te)  // K14 SiH2- + e -> SiH2  + 2e
{
  value_type K14;
  K14 = 3.15E-16 * pow((Te), 0.5) * exp(-(1.16) / (Te));
  return K14;
}

value_type k15(value_type Te)  // K15 SiH2 + e -> SiH2-
{
  value_type K15;
  K15 = 5.71E-16 * pow((Te), -0.5);
  return K15;
}

value_type k16(value_type Te)  // K16 H2 + e ->  2H + e
{
  value_type K16;
  K16 = 4.73E-14 * pow((Te), -0.23) * exp(-(10.09) / (Te));
  return K16;
}

value_type k17(value_type Te)  // K17 H2 + e ->  H2+ + 2e
{
  value_type K17;
  K17 = 1.1E-14 * pow((Te), 0.42) * exp(-(16.05) / (Te));
  return K17;
}

value_type k18(value_type Tg)  // K18 SiH4 + Ar* -> SiH3 + H + Ar
{
  value_type K18;
  K18 = 1.400e-16;
  return K18;
}

value_type k19(value_type Tg) {
  value_type K19;
  K19 = 2.591e-16;
  return K19;
}

value_type k20(value_type Tg) {
  value_type K20;
  K20 = 99.67e-18;
  return K20;
}

value_type k21(value_type Tg) {
  value_type K21;
  K21 = 9.963e-17;
  return K21;
}

value_type k22(value_type Tg) {
  value_type K22;
  K22 = 6.974e-17;
  return K22;
}

value_type k23(value_type Tg)  // k23(Tg)%K23 SiH3 + SiH3 -> SiH2 + SiH4
{
  value_type K23;
  K23 = 2.99e-17;
  return K23;
}

value_type k24(value_type Tg)  // K24 SiH4 + SIH3 -> Si2H5 + H2
{
  value_type K24;
  K24 = 2.94e-18 * exp(-0.1908 / Tg);
  return K24;
}

value_type k25(value_type Tg)  // K25 SiH2 + H2 -> SiH4
{
  value_type K25;
  K25 = 2.e-19;
  return K25;
}

value_type k26(value_type Tg)  // K26 SiH2  -> Si + H2
{
  value_type K26;
  K26 = 1.51E-9 * pow((Tg), 1.658) * exp(-(1.66) / (Tg));
  return K26;
}

value_type k27(value_type Tg)  // K27 SiH4 + H -> H2 + SiH3
{
  value_type K27;
  K27 = 2.44E-22 * pow((Tg), 1.9) * exp(-(0.09) / (Tg));
  return K27;
}

value_type k28(value_type Tg)  // K28 SiH2 + SiH2 -> Si2H2 + H2
{
  value_type K28;
  K28 = 1.08e-15;
  return K28;
}

value_type k29(value_type Tg)  // K29 SiH2 + H -> SiH + H2
{
  value_type K29;
  K29 = 2.31e-17;
  return K29;
}

value_type k30(value_type Tg)  // K30 SiH3-> SiH + H2
{
  value_type K30;
  K30 = 328.9E-6 * pow((Tg), -3.1) * exp(-(1.94) / (Tg));
  return K30;
}

value_type k31(value_type Tg)  // K31 SiH3 + H -> SiH2 + H2
{
  value_type K31;
  K31 = 2.49E-17 * exp(-(0.1084) / (Tg));
  return K31;
}

value_type k32(value_type Tg)  // K32 SiH2- + H2+ -> SiH2 + H2
{
  value_type K32;
  K32 = 5.55E-12 * pow((Tg), -0.5);
  return K32;
}

value_type k33(value_type Tg)  // K33 SiH3- + H2+ -> SiH3 + H2
{
  value_type K33;
  K33 = 5.55E-12 * pow((Tg), -0.5);
  return K33;
}

value_type k34(value_type Tg)  // K34 SiH3- +H2+ -> SiH3 + H2
{
  value_type K34;
  K34 = 2.11e-20;
  return K34;
}

value_type k35(value_type Tg)  // K35 SiH3- + SiH3+ -> Si2H6
{
  value_type K35;
  K35 = 2.11e-20;
  return K35;
}

value_type k36(value_type Tg)  // K36 SiH2- + SiH4 -> Si2H4- + H2
{
  value_type K36;
  K36 = 2.11e-20;
  return K36;
}

value_type k37(value_type Tg)  // K37 SiH2- + SiH3 -> SiH2 + SiH3-
{
  value_type K37;
  K37 = 2.11e-20;
  return K37;
}

value_type k38(value_type Tg)  // K38 SiH3- + SiH2 -> Si2H3- + H2
{
  value_type K38;
  K38 = 2.11e-20;
  return K38;
}

value_type k39(value_type Tg)  // K39 SiH3- + SiH4 -> Si2H5- + H2
{
  value_type K39;
  K39 = 2.11e-20;
  return K39;
}

value_type k40(value_type Tg)  // K40 SiH2- + SiH3+ -> Si2H5
{
  value_type K40;
  K40 = 2.11e-20;
  return K40;
}

value_type k41(value_type Tg)  // K41 SiH2- + Ar+ -> SiH2 + Ar
{
  value_type K41;
  K41 = 1.44e-12 * pow((Tg), -0.5);
  return K41;
}

value_type k42(value_type Tg)  // K42 SiH3- + Ar+ -> SiH3 + Ar
{
  value_type K42;
  K42 = 1.44E-12 * pow((Tg), -0.5);
  return K42;
}

value_type k43(value_type Te)  // K43 SiH3 + e ->  SiH3-
{
  value_type K43;
  K43 = 5.71E-16 * pow((Te), -0.5);
  return K43;
}

value_type k44(value_type Te)  // K44 SiH + e ->  SiH-
{
  value_type K44;
  K44 = 5.71E-15 * pow((Te), -0.5);
  return K44;
}

value_type k45(value_type Te)  // K45 SiH- + e ->  SiH + 2e
{
  value_type K45;
  K45 = 3.16E-16 * pow((Te), 0.5) * exp(-(1.25) / (Te));
  return K45;
}

value_type k46(value_type Tg)  // K46 SiH2- + SiH ->  SiH2 + SiH2m
{
  value_type K46;
  K46 = 2.31e-17;
  return K46;
}

value_type k47(value_type Tg)  // K47 SiH- + H2p ->  SiH + H2
{
  value_type K47;
  K47 = 3.21e-13;
  return K47;
}

struct ddriv_sys  // structure qui vient calculer les equations differentielles
{
  void operator()(int* neq, double* t, double* n, double* dndt) {
    /*0=e, 1=Armet, 2=SiH3-, 3=SiH2-, 4=SiH3+, 5=SiH4, 6=SiH3,
    7=H, 8=SiH2, 9=H2, 10=H2+, 11=Si2H5, 12=Si2H2, 13=Si2H4-,
    14=Si2H6, 15=Si2H3-, 16=Si2H5-, 17=SiH-, 18=SiH, 19=Si, 20=Arp, 21=NP*/
    value_type dpp = n[4] + n[10] + n[20];
    value_type dnn = n[0] + n[2] + n[3] + n[13] + n[15] + n[16] + n[17];
    value_type ratep =
        rate * dnn / dpp;  // pour que les charges sortent par paires = et -

    value_type n_mu =
        n[0] * mu[0] + n[4] * mu[4] + n[10] * mu[10] + n[20] * mu[20];
    value_type n_DL =
        -n[0] * DL[0] + n[4] * DL[4] + n[10] * DL[10] + n[20] * DL[20];

    value_type rr = n_DL / n_mu;
    state_type DA(Nbr_espece, 0.0);  // vecteur de diffusion ambipolaire en m2/s
    // diffusion ambipolaire
    DA[0] = DL[0] + mu[0] * rr;  // s-1
    DA[1] = DL[1];
    DA[4] = DL[4] - mu[4] * rr;
    DA[10] = DL[10] - mu[10] * rr;
    DA[20] = DL[20] - mu[20] * rr;

    value_type Vfl =
        cVfl * n[22] / n[21];  // V potentiel flottant e*qNP / 4 pi eps0 rNP
    value_type sNP = pi * pow(n[21], 2);  // surface de la NP

    // if (Vfl<0.){
    value_type facte = sNP * exp(Vfl / Te);  // qNP<0
    value_type factn = sNP * exp(Vfl / Tg);
    value_type factp = sNP * (1. - Vfl / Tg); /*}
else { facte=sNP*(1.+Vfl/Te); //else qNP>0
            factn=sNP*(1.+Vfl/Tg);
            factp=sNP*exp(-Vfl/Tg);}*/
    // calcul des coefficients pour la diffusion ambipolaire

    // chargement des NP
    value_type ffe = facte * vth[0] * n[0];
    value_type ffn2 = factn * vth[2] * n[2];
    value_type ffn3 = factn * vth[3] * n[3];
    value_type ffp4 = factp * vth[4] * n[4];
    value_type ffp10 = factp * vth[10] * n[10];
    value_type ffn13 = factn * vth[13] * n[13];
    value_type ffn15 = factn * vth[15] * n[15];
    value_type ffn16 = factn * vth[16] * n[16];
    value_type ffn17 = factn * vth[17] * n[17];
    value_type ffp20 = factp * vth[20] * n[20];

    value_type rr1 = k1(Te) * n_Ar * n[0];
    value_type rr2 = k2(Te) * n_Ar * n[0];
    value_type rr3 = k3(Te) * n[1] * n[0];
    value_type rr4 = k4(Tg) * n[1] * n[1];
    value_type rr5 = k5(Te) * n[1] * n[0];
    value_type rr6 = k6(Te) * n[5] * n[0];
    value_type rr7 = k7(Te) * n[5] * n[0];
    value_type rr8 = k8(Te) * n[5] * n[0];
    value_type rr9 = k9(Te) * n[5] * n[0];
    value_type rr10 = k10(Te) * n[5] * n[0];
    value_type rr11 = k11(Te) * n[6] * n[0];
    value_type rr12 = k12(Te) * n[6] * n[0];
    value_type rr13 = k13(Te) * n[2] * n[0];
    value_type rr14 = k14(Te) * n[3] * n[0];
    value_type rr15 = k15(Te) * n[8] * n[0];
    value_type rr16 = k16(Te) * n[9] * n[0];
    value_type rr17 = k17(Te) * n[9] * n[0];
    value_type rr18 = k18(Tg) * n[5] * n[1];
    value_type rr19 = k19(Tg) * n[5] * n[1];
    value_type rr20 = k20(Tg) * n[6] * n[1];
    value_type rr21 = k21(Tg) * n[8] * n[1];
    value_type rr22 = k22(Tg) * n[9] * n[1];
    value_type rr23 = k23(Tg) * n[6] * n[6];
    value_type rr24 = k24(Tg) * n[5] * n[6];
    value_type rr25 = k25(Tg) * n[8] * n[9];
    value_type rr26 = k26(Tg) * n[8];
    value_type rr27 = k27(Tg) * n[5] * n[7];
    value_type rr28 = k28(Tg) * n[8] * n[8];
    value_type rr29 = k29(Tg) * n[8] * n[7];
    value_type rr30 = k30(Tg) * n[6];
    value_type rr31 = k31(Tg) * n[6] * n[7];
    value_type rr32 = k32(Tg) * n[3] * n[10];
    value_type rr33 = k33(Tg) * n[2] * n[10];
    value_type rr34 = k34(Tg) * n[2] * n[6];
    value_type rr35 = k35(Tg) * n[2] * n[4];
    value_type rr36 = k36(Tg) * n[3] * n[5];
    value_type rr37 = k37(Tg) * n[3] * n[6];
    value_type rr38 = k38(Tg) * n[2] * n[8];
    value_type rr39 = k39(Tg) * n[2] * n[5];
    value_type rr40 = k40(Tg) * n[3] * n[4];
    value_type rr41 = k41(Tg) * n[3] * n[20];
    value_type rr42 = k42(Tg) * n[2] * n[20];
    value_type rr43 = k43(Te) * n[6] * n[0];
    value_type rr44 = k44(Te) * n[18] * n[0];
    value_type rr45 = k45(Te) * n[17] * n[0];
    value_type rr46 = k46(Tg) * n[3] * n[18];
    value_type rr47 = k47(Tg) * n[17] * n[10];

    value_type ccol = nNP * 4. * sNP / vSi;  // 4*sNP=surface de la NP

    dndt[0] = rr1 + rr3 + rr4 - rr8 - rr9 + rr10 - rr11 + rr12 + rr13 + rr14 -
              rr15 + rr17 - rr43 - rr44 + rr45 - DA[0] * n[0] -
              rate * n[0]  /*! sortie pompe*/
              - ffe * nNP; /*! chargement des NP  */

    dndt[1] = rr2 - rr3 - 2. * rr4 - rr5 - rr18 - rr19 - rr20 - rr21 - rr22 -
              DA[1] * n[1] - rate * n[1]   /*! sortie pompe*/
              - sNP * vth[1] * nNP * n[1]; /* ! pertes sur les NP*/

    dndt[2] = rr8 - rr13 - rr33 - rr34 - rr35 + rr37 - rr38 - rr39 - rr42 +
              rr43 - rate * n[2] /*! sortie pompe*/
              - ffn2 * nNP;      /*! chargement des NP*/

    dndt[3] = rr9 + rr11 - rr14 + rr15 - rr32 - rr36 - rr37 - rr40 - rr41 -
              rr46 - rate * n[3] /*! sortie pompe*/
              - ffn3 * nNP;      /* ! chargement des NP*/

    dndt[4] = rr10 + rr12 - rr35 - rr40 - DA[4] * n[4] -
              ratep * n[4]  /*! sortie pompe*/
              - ffp4 * nNP; /* ! chargement des NP*/

    dndt[5] = C /*   ! taux d'injection du SiH4 dans le reacteur*/
              - rr6 - rr7 - rr8 - rr9 - rr10 - rr18 - rr19 + rr23 - rr24 +
              rr25 - rr27 - rr36 - rr39 - rate * n[5] /*! sortie pompe*/
              - ccol * coll[5] * n[5];                /*! collage sur les NP*/

    dndt[6] = rr6 - rr11 - rr12 + rr13 + rr18 - rr20 - 2. * rr23 - rr24 + rr27 -
              rr30 - rr31 + rr33 - rr34 - rr37 + rr42 - rr43 -
              rate * n[6]              /* ! sortie pompe*/
              - ccol * coll[6] * n[6]; /*! collage sur les NP*/

    dndt[7] = rr6 + 2. * rr7 + rr8 + 2. * rr9 + rr10 + rr11 + 2. * rr16 + rr18 +
              2. * rr19 + rr20 + rr21 + 2. * rr22 - rr27 - rr29 - rr31 -
              rate * n[7] +
              (ffn2 + ffp4 + ffn15 + ffn16 + ffn17) *
                  nNP /*! liberation de H lors du chargement des NP*/
              + ccol * (coll[6] * n[6] + coll[11] * n[11] + coll[18] * n[18]);

    dndt[8] = rr7 + rr14 - rr15 + rr19 + rr20 - rr21 + rr23 - rr25 - rr26 -
              2. * rr28 - rr29 + rr31 + rr32 + rr37 - rr38 + rr41 + rr46 -
              rate * n[8] - ccol * coll[8] * n[8]; /* ! collage sur les NP*/

    dndt[9] = -rr16 - rr17 - rr22 + rr24 - rr25 + rr26 + rr27 + rr28 + rr29 +
              rr30 + rr31 + rr32 + rr33 + rr34 + rr36 + rr38 + rr39 + rr47 -
              rate * n[9] + DA[10] * n[10] /*! H2+ + e -> H2 par diffusion*/
              + (ffn2 + ffn3 + ffp4 + ffp10 + 2. * ffn13 + ffn15 + 2. * ffn16) *
                    nNP /* ! liberation de H2 lors du chargement des NP*/
              + ccol * (2. * coll[5] * n[5] + coll[6] * n[6] + coll[8] * n[8] +
                        2. * coll[11] * n[11] + coll[12] * n[12] +
                        3. * coll[14] * n[14]) +
              DA[4] * n[4];
    /*!  SiH3+ + e -> SiH + H2, Si + H2 + H par diffusion
     * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

    dndt[10] = rr17 - rr32 - rr33 - rr47 - DA[10] * n[10] - ratep * n[10] -
               ffp10 * nNP; /* ! chargement des NP*/

    dndt[11] = rr24 + rr40 - rate * n[11] -
               ccol * coll[11] * n[11]; /* ! collage sur les NP*/

    dndt[12] = rr28 - rate * n[12] -
               ccol * coll[12] * n[12]; /*    ! collage sur les NP*/

    dndt[13] =
        rr34 + rr36 - rate * n[13] - ffn13 * nNP; /*  ! chargement des NP*/

    dndt[14] = rr35 - rate * n[14] -
               ccol * coll[14] * n[14]; /*    ! collage sur les NP*/

    dndt[15] = rr38 - rate * n[15] - ffn15 * nNP; /* !  chargement des NP*/

    dndt[16] = rr39 - rate * n[16] - ffn16 * nNP; /*   ! chargement des NP*/

    dndt[17] = rr44 - rr45 + rr46 - rr47 - rate * n[17] -
               ffn17 * nNP; /* ! chargement des NP*/

    dndt[18] = rr21 + rr29 + rr30 - rr44 + rr45 - rr46 + rr47 - rate * n[18] -
               ccol * coll[18] * n[18] /*  ! collage sur les NP*/
               + DA[4] * n[4]; /*  !  SiH3+ + e -> SiH + H2 par diffusion
                                  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

    dndt[19] =
        rr26 - rate * n[19] - ccol * coll[19] * n[19]; /* ! collage sur les NP*/

    dndt[20] = rr1 + rr3 + rr4 - rr41 - rr42 - DA[20] * n[20] -
               ratep * n[20]  /*        ! sortie*/
               - ffp20 * nNP; /*   ! chargement des NP*/

    /*c Collage dR/dt sur les NP:*/

    dndt[21] =
        coll[5] * n[5] + coll[6] * n[6] + coll[8] * n[8] +
        2. * coll[11] * n[11] + 2. * coll[12] * n[12] + +2. * coll[14] * n[14] +
        coll[18] * n[18] + coll[19] * n[19] +
        (ffn2 + ffn3 + 2. * ffn13 + 2. * ffn15 + 2. * ffn16 + ffn17 + ffp4) *
            vSi / sNP;

    /*c Charge/e des NP par OML:*/

    dndt[22] = /* ! s-1 ! dQN/dt*/
        -ffe - ffn2 - ffn3 - ffn13 - ffn15 - ffn16 - ffn17 + ffp4 + ffp10 +
        ffp20;

    /*c Densite des NP:*/

    dndt[23] = dndt[13] + dndt[16];
  }
  state_type n;
  value_type Te;
  value_type DP;
  int p1, p2, g1, g2, g3, g4;
  value_type Tp, Tx, Tj;
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
  value_type operator()(value_type const& Te)

  {  // calcul des coefficients pour la diffusion ambipolaire
    value_type n_mu =
        mu[0] * n[0] + n[4] * mu[4] + n[10] * mu[10] + n[20] * mu[20];
    value_type n_DL =
        -n[0] * DL[0] + n[4] * DL[4] + n[10] * DL[10] + n[20] * DL[20];

    // diffusion ambipolaire
    value_type Diffe = DL[0] + mu[0] * n_DL / n_mu;  // s-1

    value_type Vfl =
        cVfl * n[22] / n[21];  // V potentiel flottant e*qNP / 4 pi eps0 rNP
    value_type sNP = pi * pow(n[21], 2);  // surface de la NP
    // if (Vfl<=0.){
    value_type Fe = sNP * vth[0] * exp(Vfl / Te);
    /*}
else { Fe=sNP*vth[0]*(1.-Vfl/Te);}*/

    return -DP / Te + k1(Te) * n_Ar * 15.76 + k2(Te) * n_Ar * 11.76 +
           k3(Te) * n[1] * 4. - k5(Te) * n[1] * 11.76 + k6(Te) * n[5] * 10.68 +
           k7(Te) * n[5] * 10.68 + k8(Te) * n[5] * 8.29 + k9(Te) * n[5] * 8.29 +
           k10(Te) * n[5] * 12 + k11(Te) * n[6] * 1.94 + k12(Te) * n[6] * 1.30 +
           k13(Te) * n[2] * 1.16 + k14(Te) * n[3] * 1.16 +
           k15(Te) * n[8] * 1.5 * Te + k16(Te) * n[9] * 10.09 +
           k17(Te) * n[9] * 16.05 + k43(Te) * n[6] * 1.5 * Te +
           k44(Te) * n[18] * 1.5 * Te + k45(Te) * n[17] * 1.25 +
           Diffe * 1.5 * Te      /*perte sur les parois*/
           + Fe * nNP * 1.5 * Te /*pertes sur les NP*/
           + rate * 1.5 *
                 Te; /*perte taux injection evacuation


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
  value_type DP;
  // fonction pour calculer les K en faisant
  // varier Te dans la bissection

  // Tab as member
};

// ecriture des densite dans un fichier

void write_density(ofstream& fp, const value_type t, const value_type Te,
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
  outfile.open("densnonautom.dat");

  value_type Te = 3.;  // valeur initiale de la temperature

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

  value_type cvth = 1.56e4 * sqrt(Tg);  // NRL * sqrt (8/pi)

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

  value_type DP = 0.5 * DL[0] * pow((V / L), 2.);

  for (int i = 0; i < Nbr_espece; i++) {
    DL[i] = DL[i] * diff;
    mu[i] = mu[i] * diff;
  }

  // variable du temps
  double t = 0.0;
  value_type dt = 1.0e-8;
  value_type Tmax = 1.0;//20.e-3;
  value_type NT = Tmax / dt;

  // variable pour la bissection
  value_type min = Tg;
  value_type max = 100.;
  boost::uintmax_t max_iter = 100;
  eps_tolerance<value_type> tol(10);

  state_type n_new(Nbr_espece, 0.0);  // initialisation du vecteur densite
  n_new = n_ini;
  state_type n_err(Nbr_espece, 0.0);  // error

  // declare la fonction etemperature
  etemperature etemp;

  // assigne les valeur a la fonction etemp
  etemp.n = n_ini;

  // assigne les vecteurs Dl  mu et vth a la fonction etemp
  etemp.DL = DL;
  etemp.mu = mu;
  etemp.vth = vth;

  // assigne DP a etemp
  etemp.DP = DP;

  // premier calcul de Te
  pair<value_type, value_type> pair_Te =
      toms748_solve(etemp, min, max, tol, max_iter);

  Te = pair_Te.first;
  cerr << "\n[ii]  Temperature Initiale = " << Te << endl;

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
    mu[1] = DL[1] / Te;          // 30./(pression*760.)     m2/(V.s)
    vth[1] = 6.69e5 * sqrt(Te);  // m/s  vitesse thermique electrons

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
