from __future__ import annotations

from dataclasses import dataclass

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.mesh import Mesh, MeshTags

from preproc.config import Config, ContactBoundary, NeumannBoundary
from preproc.contact import contact_traction
from preproc.materials import Material, build_material
from preproc.neumann import neumann_traction
from utils import Expr, JIT_OPTIONS, SCALAR, owned_size


def _displacement_part(f: Expr, mixed: bool) -> Expr:
    """Partie déplacement d'une expression UFL de l'espace de l'inconnue (u de (u, p) si mixte)."""
    return ufl.split(f)[0] if mixed else f


@dataclass
class Forms:
    """
    Formulation variationnelle prête à être résolue par solve_newton.

    Deux cas, selon la loi de comportement (voir preproc/materials.py) :
      * élasticité linéaire : V est l'espace vectoriel du déplacement ;
      * hyperélasticité     : V est l'espace mixte (u, p) de Taylor-Hood P_k - P_(k-1).
    Les méthodes d'accès ci-dessous masquent cette différence aux autres modules : conditions aux
    limites, réactions et post-traitement n'ont jamais à savoir si V est mixte ou non.
    """
    V: fem.FunctionSpace       # espace de fonctions de l'inconnue (vectoriel, ou mixte (u, p))
    uh: fem.Function           # inconnue, modifiée en place par le solveur
    residual: fem.Form         # forme compilée R(uh; v) = f_int - f_ext
    jacobian: fem.Form         # forme compilée dR/d(uh) (tangente)
    external: fem.Form         # forme compilée des forces extérieures f_ext : charges pilotées (x lambda)
                                # + réaction de contact (non mise à l'échelle, voir preproc/contact.py) ;
                                # postproc/reactions.py s'appuie sur cette somme complète pour le bilan global
    load: fem.Constant         # facteur de charge lambda dans [0, 1] (gravité + Neumann)
    dx: ufl.Measure            # mesure de volume (degré de quadrature inclus)
    mixed: bool                # True : formulation mixte u-p
    material: Material         # loi de comportement (contraintes, déformations et J pour le post-traitement)

    # --- accès au déplacement ------------------------------------------------
    def component_space(self, comp: int) -> fem.FunctionSpace:
        """Sous-espace scalaire de la composante 'comp' du déplacement (vue sur V)."""
        return self.V.sub(0).sub(comp) if self.mixed else self.V.sub(comp)

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
        if not self.mixed:
            return self.uh
        u = self.uh.sub(0).collapse()
        u.name = "displacement"
        return u

    # --- accès symbolique (expressions UFL) -----------------------------------
    def displacement_ufl(self) -> Expr:
        """Déplacement comme expression UFL (u de (u, p) en formulation mixte)."""
        return _displacement_part(self.uh, self.mixed)

    def cauchy_ufl(self) -> Expr:
        """Contraintes de Cauchy à l'état courant, comme expression UFL."""
        if self.mixed:
            u, p = ufl.split(self.uh)
            return self.material.cauchy(u, p)
        return self.material.cauchy(self.uh)


# -----------------------------------------------------------------------------
# Formulation variationnelle
#   déplacement seul : R(u; v)       = ∫ P(u):grad(v) dx - W_ext(v)
#   mixte (u, p)      : R(u, p; w)   = dΠ(u, p)[w] - W_ext(v),   Π = ∫ ψ_iso(F) + p (J-1) - p²/(2K) dx
#   W_ext = lambda * (travail des charges) + W_contact(v)
# Le travail interne est fourni par la loi (Material.internal_work) : pas de branche par famille ici
# hormis la construction de l'espace de fonctions. Les règles de compatibilité (mesh.order >= 2 pour la
# formulation mixte, etc.) sont vérifiées par la config (Material.check_setup).
# Le contact (preproc/contact.py) ajoute un travail virtuel W_contact(v) = ∫_Γc t_c(uh)·v ds, NON
# multiplié par lambda (voir boucle ci-dessous) : c'est une réaction de contrainte unilatérale, pas une
# charge pilotée. Comme t_c dépend de uh (jauge g_N et glissement tangentiel), sa contribution à la
# tangente est obtenue par ufl.derivative exactement comme le reste du résidu : aucune infrastructure
# supplémentaire n'est nécessaire dans solver/solver.py.
# -----------------------------------------------------------------------------
def build_forms(
    domain: Mesh, facet_tags: MeshTags, boundary_tags: dict[str, int], config: Config
) -> Forms:
    gdim = domain.geometry.dim
    order = config.mesh.order
    cell = domain.topology.cell_name()

    # 1. Loi de comportement (choisie dans le YAML) : elle impose la formulation
    material = build_material(domain, config.material.params())
    mixed = material.MIXED

    # 2. Espace de fonctions et inconnue
    u_element = basix.ufl.element("Lagrange", cell, order, shape=(gdim,))
    if mixed:
        p_element = basix.ufl.element("Lagrange", cell, order - 1)
        V = fem.functionspace(domain, basix.ufl.mixed_element([u_element, p_element]))
    else:
        V = fem.functionspace(domain, u_element)

    uh = fem.Function(V, name="solution" if mixed else "displacement")
    w = ufl.TestFunction(V)
    dw = ufl.TrialFunction(V)
    v = _displacement_part(w, mixed)

    # 3. Mesures d'intégration (le degré de quadrature ne porte que sur le volume)
    quad = config.mesh.quadrature_degree
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quad} if quad is not None else {})
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    # 4. Charges extérieures, pilotées par le facteur lambda (chargement proportionnel)
    load = fem.Constant(domain, SCALAR(0.0))
    rho = config.material.rho
    f_body = np.array([rho * g for g in config.physics.gravity], dtype=SCALAR)
    f = fem.Constant(domain, f_body)

    W_ext = load * ufl.inner(f, v) * dx
    for bnd in config.boundaries:
        if isinstance(bnd, NeumannBoundary):
            tag = boundary_tags[bnd.name]
            traction = fem.Constant(domain, neumann_traction(domain, ds, tag, bnd))
            W_ext += load * ufl.inner(traction, v) * ds(tag)

    # 4bis. Contact unilatéral (obstacle rigide analytique) : réaction de contrainte, jamais mise à
    # l'échelle par lambda (voir le commentaire au-dessus de build_forms).
    X = ufl.SpatialCoordinate(domain)
    u_expr = _displacement_part(uh, mixed)
    for bnd in config.boundaries:
        if isinstance(bnd, ContactBoundary):
            tag = boundary_tags[bnd.name]
            t_c = contact_traction(u_expr, X + u_expr, bnd, gdim)
            W_ext += ufl.inner(t_c, v) * ds(tag)  # pas de 'load *' : réaction de contrainte, pas une charge pilotée

    # 5. Résidu, tangente par dérivation automatique, puis compilation
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
        mixed=mixed,
        material=material,
    )
