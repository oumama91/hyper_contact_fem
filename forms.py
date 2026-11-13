from __future__ import annotations

from dataclasses import dataclass

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.mesh import Mesh, MeshTags

from config import Config, ContactBoundary
from contact import contact_traction
from neo_hookean import NeoHookean
from utils import Expr, JIT_OPTIONS, SCALAR, owned_size


@dataclass
class Forms:
    """
    Formulation variationnelle mixte (u, p) prête à être résolue par solve_newton.

    V est l'espace mixte de Taylor-Hood P_k - P_(k-1). Les méthodes d'accès ci-dessous
    masquent l'imbrication (u, p) aux autres modules : conditions aux limites, réactions et
    post-traitement n'ont jamais à manipuler l'inconnue mixte.
    """
    V: fem.FunctionSpace       # espace de fonctions mixte (u, p)
    uh: fem.Function           # inconnue, modifiée en place par le solveur
    residual: fem.Form         # forme compilée R(uh; v) = f_int - f_ext
    jacobian: fem.Form         # forme compilée dR/d(uh) (tangente)
    external: fem.Form         # forme compilée des forces extérieures f_ext : gravité pilotée (x lambda)
                                # + réaction de contact (non mise à l'échelle, voir contact.py) ;
                                # reactions.py s'appuie sur cette somme complète pour le bilan global
    load: fem.Constant         # facteur de charge lambda dans [0, 1] (gravité)
    dx: ufl.Measure            # mesure de volume (degré de quadrature inclus)
    material: NeoHookean       # loi de comportement (contraintes, déformations et J pour le post-traitement)

    # --- accès au déplacement ------------------------------------------------
    def component_space(self, comp: int) -> fem.FunctionSpace:
        """Sous-espace scalaire de la composante 'comp' du déplacement (vue sur V)."""
        return self.V.sub(0).sub(comp)

    def component_dofs(self, comp: int, facets: np.ndarray) -> np.ndarray:
        """Dofs (numérotation de V) de la composante 'comp' du déplacement sur les facettes données."""
        fdim = self.V.mesh.topology.dim - 1
        return fem.locate_dofs_topological(self.component_space(comp), fdim, facets)

    def owned_component_dofs(self, comp: int) -> np.ndarray:
        """Dofs (numérotation de V) de la composante 'comp' du déplacement possédés par ce processus."""
        n_flat = owned_size(self.V)
        _, parent = self.component_space(comp).collapse()
        parent = np.asarray(parent, dtype=np.int64)
        return parent[parent < n_flat]

    def displacement(self) -> fem.Function:
        """Champ de déplacement (Function dans un espace vectoriel autonome)."""
        u = self.uh.sub(0).collapse()
        u.name = "displacement"
        return u

    # --- accès symbolique (expressions UFL) -----------------------------------
    def displacement_ufl(self) -> Expr:
        """Déplacement comme expression UFL (u de (u, p))."""
        return ufl.split(self.uh)[0]

    def cauchy_ufl(self) -> Expr:
        """Contraintes de Cauchy à l'état courant, comme expression UFL."""
        u, p = ufl.split(self.uh)
        return self.material.cauchy(u, p)


# -----------------------------------------------------------------------------
# Formulation variationnelle mixte
#   R(u, p; w) = dΠ(u, p)[w] - W_ext(v),   Π = ∫ ψ_iso(F) + p (J-1) - p²/(2K) dx
#   W_ext = lambda * (travail de la gravité) + W_contact(v)
# Le travail interne est fourni par la loi (NeoHookean.internal_work) ; les règles de
# compatibilité (mesh.order >= 2 pour Taylor-Hood) sont vérifiées par la configuration.
# Le contact (contact.py) ajoute un travail virtuel W_contact(v) = ∫_Γc t_c(uh)·v ds,
# NON multiplié par lambda : c'est une réaction de contrainte unilatérale, pas une charge
# pilotée. Comme t_c dépend de uh (jauge g_N et glissement tangentiel), sa contribution à la
# tangente est obtenue par ufl.derivative exactement comme le reste du résidu.
# -----------------------------------------------------------------------------
def build_forms(
    domain: Mesh, facet_tags: MeshTags, boundary_tags: dict[str, int], config: Config
) -> Forms:
    gdim = domain.geometry.dim
    order = config.mesh.order
    cell = domain.topology.cell_name()

    # 1. Loi de comportement : néo-hookéen mixte (seule loi du projet).
    material = NeoHookean(domain, config.material.params())

    # 2. Espace de fonctions mixte et inconnue (Taylor-Hood P_k - P_(k-1)).
    u_element = basix.ufl.element("Lagrange", cell, order, shape=(gdim,))
    p_element = basix.ufl.element("Lagrange", cell, order - 1)
    V = fem.functionspace(domain, basix.ufl.mixed_element([u_element, p_element]))
    uh = fem.Function(V, name="solution")
    w = ufl.TestFunction(V)
    dw = ufl.TrialFunction(V)
    v = ufl.split(w)[0]

    # 3. Mesures d'intégration (le degré de quadrature ne porte que sur le volume)
    quad = config.mesh.quadrature_degree
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quad} if quad is not None else {})
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    # 4. Charges extérieures pilotées par le facteur lambda (gravité uniquement)
    load = fem.Constant(domain, SCALAR(0.0))
    rho = config.material.rho
    f_body = np.array([rho * g for g in config.physics.gravity], dtype=SCALAR)
    f = fem.Constant(domain, f_body)

    W_ext = load * ufl.inner(f, v) * dx

    # 5. Contact unilatéral (obstacle rigide analytique) : réaction de contrainte, jamais mise à
    # l'échelle par lambda (voir le commentaire au-dessus de build_forms).
    X = ufl.SpatialCoordinate(domain)
    u_expr = ufl.split(uh)[0]
    for bnd in config.boundaries:
        if isinstance(bnd, ContactBoundary):
            tag = boundary_tags[bnd.name]
            t_c = contact_traction(u_expr, X + u_expr, bnd, gdim)
            W_ext += ufl.inner(t_c, v) * ds(tag)  # pas de 'load *' : réaction de contrainte, pas une charge pilotée

    # 6. Résidu, tangente par dérivation automatique, puis compilation
    R = material.internal_work(uh, w, dx) - W_ext
    Jac = ufl.derivative(R, uh, dw)

    return Forms(
        V=V,
        uh=uh,
        residual=fem.form(R, jit_options=JIT_OPTIONS),
        jacobian=fem.form(Jac, jit_options=JIT_OPTIONS),
        external=fem.form(W_ext, jit_options=JIT_OPTIONS),
        load=load,
        dx=dx,
        material=material,
    )
