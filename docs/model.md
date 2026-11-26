# Modèle mécanique

## Périmètre

Le projet traite un seul corps déformable, une seule loi et un seul lecteur de maillage. Il ne
fait pas :

- contact **corps-à-corps** : un seul solide, l'obstacle est analytique ;
- chargement de **Neumann** (pression, effort surfacique) : seule la gravité est pilotée ;
- autre loi que le **néo-hookéen mixte** (ni viscoélasticité, ni plasticité) ;
- dynamique, génération ni adaptation de maillage.

Système cohérent **mm, t, s, N, MPa** (1 t accélérée à 1 mm/s² donne 1 N ; 1 N/mm² = 1 MPa) :
longueurs en mm, temps en s, masse en t, contrainte en MPa, $\rho$ en t/mm³, gravité en mm/s²,
$\varepsilon$ en MPa/mm.

Avec $\mathbf F = \mathbf I + \nabla\mathbf u$, $J = \det\mathbf F$ et $\mathbf w = (\mathbf v,q)$
un champ test mixte :

$$\Pi(\mathbf u,p) = \int_{\Omega_0}\left[\psi_{\text{iso}}(\mathbf F) + p\,(J-1) - \frac{p^2}{2K}\right]\mathrm d\Omega_0 - W_{\text{ext}}(\mathbf v),\qquad \psi_{\text{iso}} = \frac{\mu}{2}\left(\bar I_1-3\right)$$

$$\bar I_1 = J^{-2/3}\,\mathrm{tr}(\mathbf F^T\mathbf F),\quad \mu = \frac{E}{2(1+\nu)},\quad K = \frac{E}{3(1-2\nu)}$$

La stationnarité en $p$ donne $p = K(J-1)$ : $p$ est une contrainte moyenne (positive en
traction), pas une pression. $E$ [MPa] et $\nu$ [-] sont les paramètres matériau ; $\nu = 0{,}5$
donne $1/K = 0$ (incompressibilité stricte), le cas de référence prend $\nu = 0{,}4999$.

$$\mathbf P = \frac{\partial\psi}{\partial\mathbf F} = \mathbf P_{\text{iso}} + p\,J\,\mathbf F^{-T},\quad \boldsymbol\sigma = \frac{1}{J}\mathbf P\mathbf F^T = \frac{1}{J}\mathbf P_{\text{iso}}\mathbf F^T + p\,\mathbf I,\quad \mathbf E = \tfrac{1}{2}(\mathbf F^T\mathbf F-\mathbf I)$$

$\mathbf P$ est le premier tenseur de Piola-Kirchhoff (PK1), $\boldsymbol\sigma$ la contrainte de
Cauchy, $\mathbf E$ la déformation de Green-Lagrange, $\lambda_i = \sqrt{1+2E_i}$ les allongements
principaux. La partie isochore de $\boldsymbol\sigma$ étant de trace nulle, $\mathrm{tr}(\boldsymbol\sigma)/3 = p$ (champ `mean_stress`).

La position courante est $\mathbf x = \mathbf X + \mathbf u$. Chaque obstacle fournit une jauge
signée $g_N(\mathbf x)$ et une normale sortante $\mathbf n$ (côté admissible) : $g_N > 0$ hors
contact, $g_N < 0$ en pénétration. La pression normale est pénalisée, $p_N = \varepsilon_n\langle -g_N\rangle$
avec $\langle a\rangle = \max(a,0)$ ; la traction tangentielle est un retour radial sur le cône de Coulomb :

$$\mathbf u_t = (\mathbf I-\mathbf n\otimes\mathbf n)\,\mathbf u,\quad \mathbf t_T = -\varepsilon_t\mathbf u_t\min\left(1,\frac{\mu_f\,p_N}{\lVert\varepsilon_t\mathbf u_t\rVert_\eta}\right),\quad \lVert\mathbf t\rVert_\eta = \sqrt{\mathbf t\cdot\mathbf t+\eta^2},\quad \eta = 10^{-9}\varepsilon_t$$

$$\mathbf t_c = p_N\,\mathbf n + \mathbf t_T,\quad W_{\text{ext}}(\mathbf v) = \lambda\int_{\Omega_0}\rho\,\mathbf g\cdot\mathbf v\,\mathrm d\Omega_0 + \int_{\Gamma_c}\mathbf t_c\cdot\mathbf v\,\mathrm ds$$

$\mu_f$ est le coefficient de frottement. La traction de contact n'est **pas** multipliée par
$\lambda$ (réaction de contrainte, active dès $g_N < 0$) ; l'intégrale porte sur la **facette de
référence** (surface non réactualisée) et le frottement holonome suppose un chargement monotone.

Le résidu et sa tangente viennent de la dérivation automatique UFL. $\lambda$ va de 0 à 1 par
incréments ($1/$`increments`) ; seuls Dirichlet et gravité sont multipliés par $\lambda$. Les
Dirichlet sont imposées par **relèvement** (`apply_lifting`), ce qui répartit le saut dans tout le
domaine. La convergence porte sur le résidu **libre** $r_k \le \max(r_{\text{tol}}\,r_{\text{ref}}, a_{\text{tol}})$,
où $r_{\text{ref}}$ est la norme du premier résidu libre de l'incrément. Chaque correction est
essayée avec un pas $\alpha = 1, 1/2, \ldots, 2^{-6}$, accepté si le résidu libre décroît
strictement ; sinon l'état est restauré et le pas réduit de moitié, l'incrément étant retenté au
plus `max_cutbacks` fois. Après deux incréments faciles, le pas est agrandi d'un facteur 1,5
jusqu'au pas nominal. La tangente étant de type point-selle indéfinie, le solveur est un **LU
direct MUMPS** (`preonly`/`lu`/`mumps`).

## Limites connues

- **Un seul corps** : contact unilatéral contre un obstacle analytique, pas de corps-à-corps.
- **Pénalité sans multiplicateurs** : non-pénétration approchée, pénétration $\propto 1/\varepsilon_n$.
- **Frottement holonome** : chargement monotone seulement ; un cycle exigerait une variable d'état.
- **Indicateur** : `sigma_1_max` est une traction maximale ; la compression minimale serait plus parlante.
- **Axe du cylindre et centre de la sphère** : la normale divise par la distance radiale (`d_perp/r` au
  cylindre, `d/r` à la sphère) ; en `r = 0` elle est indéfinie,
  `hyper_contact_fem/contact.py` ne protège pas cette division et ce cas n'est pas traité.
- **Solveur et chargement** : LU MUMPS (tailles modérées) ; $\lambda$ ne multiplie que gravité et
  Dirichlet, donc un cas porté par le seul contact est indépendant de $\lambda$.
