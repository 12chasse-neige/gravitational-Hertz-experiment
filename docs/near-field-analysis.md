> **Implementation status:** The finite-distance equations in this report now
> underpin production. Its original diagnostic workflow and implementation
> checklist and the 3995 m numerical defaults below are historical, preceding
> the CE2 silicon switch. See [the integration review guide](production-integration-review.md)
> and [current commands](current-workflows.md) for the active 40 km detector.

# Finite-distance rotor gravity and the interferometer response

The near-zone omission changes the interpretation and the predicted signal substantially. At the paper benchmark, a conserved quadrupole calculation gives an ideal free-mass Michelson response of $1.936\times10^{-32}$ at 600 Hz, compared with $2.313\times10^{-37}$ from the existing radiative-only calculation at the same geometry. The enhancement is about 83,700 in amplitude. It is predominantly the oscillating Newtonian tidal interaction. Phase alignment remains possible, provided it aligns each source's complete complex detector response.

There is also a useful geometry change: placing the source on the extension of the $+y$ arm, 2 km beyond its end mirror, with the rotor axis along $+z$, gives $2.026\times10^{-30}$ for the same rotor and 6 km source-to-vertex distance. Using the repository's ideal detuned quantum-noise curve as a conditional strain-noise proxy gives a one-year single-source SNR of 1.22, versus 0.0117 at the old geometry. These are ideal model results, not forecasts including the actual suspension, feedback, optical losses, environmental coupling, and rotor engineering.

The calculation below is first order in $G$, at leading slow-motion mass-quadrupole order, with no expansion in $kR$, $kL$, or $L/R$. Higher source multipoles are estimated and independently checked against direct Newtonian integration over the finite holes. The report does not claim an exact elastic-source solution or a completed model of a built interferometer.

## 1. Three different approximation parameters

Use $\omega_{\rm rot}$ for the rotor spin angular frequency and

\[
\Omega=2\omega_{\rm rot},\qquad f_0=\frac{\Omega}{2\pi}=600\ {\rm Hz},
\qquad k=\frac{\Omega}{c},\qquad \lambda=\frac{2\pi}{k}.
\tag{1}
\]

For the explicit paper benchmark $L=4000$ m, $R=6000$ m, and the repository's $c=2.998\times10^8$ m/s,

\[
kR=0.0754485,\qquad kL=0.0502990,\qquad L/R=2/3,
\qquad k^{-1}=79.5244\ {\rm km}.
\tag{2}
\]

The radius of a sphere enclosing the rotor is $a=\sqrt{(D/2)^2+(H/2)^2}=2.693$ m. Thus

\[
a/R=4.49\times10^{-4},\qquad ka=3.39\times10^{-5},\qquad
v_{\rm rim}/c=1.57\times10^{-5}.
\tag{3}
\]

These facts have different consequences:

| Parameter | What it controls | Status here |
|---|---|---|
| $a/r$, $ka$ | Truncation of the source-size multipole expansion | Small throughout both arms |
| $kr$ | Whether near-zone or radiative radial terms dominate | Near zone |
| $L/R$ | Whether the field is spatially uniform across the detector | Not small |
| $kL$ | Light-travel-time correction over one arm | Small, but retained below |

The $Q/r^3$, $\dot Q/(cr^2)$, and $\ddot Q/(c^2r)$ terms are different radial pieces of the same mass quadrupole. They are not respectively quadrupole, octupole, and higher multipoles. The severe error is dropping near-zone pieces at fixed multipole order. It does not imply that an unlimited number of source multipoles become important when $kR\ll1$.

The manuscript's local TT projector cannot repair the omission. A position-dependent algebraic projection of $h_{ij}$ is not a gauge transformation of the full spacetime metric. Also, $R\gg L$ alone is insufficient to justify a radiative TT field: the source-to-detector field must additionally be in its wave zone, $kR\gg1$.

## 2. Start from a conserved stress-energy tensor

Adopt signature $(-+++)$, $x^0=ct$, and

\[
g_{\mu\nu}=\eta_{\mu\nu}+h_{\mu\nu},\quad
\bar h_{\mu\nu}=h_{\mu\nu}-\tfrac12\eta_{\mu\nu}h,\quad
\partial_\mu\bar h^{\mu\nu}=0.
\tag{4}
\]

Linearized GR gives

\[
\Box\bar h^{\mu\nu}=-\frac{16\pi G}{c^4}T^{\mu\nu},\qquad
\bar h^{\mu\nu}(t,\mathbf x)=\frac{4G}{c^4}\int
\frac{T^{\mu\nu}(t-|\mathbf x-\mathbf y|/c,\mathbf y)}{|\mathbf x-\mathbf y|}\,d^3y.
\tag{5}
\]

Here $T^{\mu\nu}$ must obey $\partial_\mu T^{\mu\nu}=0$ to the retained order. It includes the stresses supplying centripetal acceleration and, when relevant, the supports, drive, and reaction system. Prescribed masses moving in circles with only $T^{ij}=\rho v_i v_j$ are not a conserved isolated source.

Although stresses are small compared with $T^{00}\sim\rho c^2$, they need not be small compared with $T^{ij}\sim\rho v_i v_j$. Omitting them can spoil the leading spatial radiative field. For the ideal steadily rotating, compact source, conservation identities let us derive the leading mass-quadrupole field without first solving every elastic stress component. Actual time-dependent support motion would supply additional moments and requires its own model.

## 3. Expand source size, retaining all propagation dependence

Choose the source center of mass as the origin for the following field derivation, and define $r=\vert \mathbf x\vert >a$. For a Fourier component use peak phasors:

\[
X(t)=\Re[\mathsf X e^{-i\Omega t}].
\tag{6}
\]

Equation (5) then has the spatial kernel $g_k(\mathbf x)=e^{ikr}/r$. Taylor expansion in source coordinates gives

\[
\frac{e^{ik|\mathbf x-\mathbf y|}}{|\mathbf x-\mathbf y|}
=\sum_{\ell=0}^{\infty}\frac{(-1)^\ell}{\ell!}
y_{i_1}\cdots y_{i_\ell}\partial_{i_1}\cdots\partial_{i_\ell}g_k(\mathbf x),
\tag{7}
\]

\[
\bar{\mathsf h}^{\mu\nu}=\frac{4G}{c^4}
\sum_{\ell=0}^{\infty}\frac{(-1)^\ell}{\ell!}\partial_Lg_k
\int \mathsf T^{\mu\nu}(\mathbf y)y_L\,d^3y.
\tag{8}
\]

The spatial derivatives act on both $1/r$ and $e^{ikr}$. We have not replaced them by $-n_i\partial_t/c$, the replacement that would discard the near-zone terms. Traces of higher raw moments can be regrouped into STF moments with finite-source retardation corrections; at leading slow-motion order these corrections are controlled by $ka$.

To connect the three metric sectors, define the ordinary second mass moment

\[
I_{ij}=\frac1{c^2}\int T^{00}y_i y_j\,d^3y,
\qquad Q_{ij}=I_{ij}-\tfrac13\delta_{ij}I_{aa}.
\tag{9}
\]

Integrating the conservation equations by parts, with localized-source boundary terms vanishing,

\[
\dot I_{ij}=\frac1c\int(T^{0i}y_j+T^{0j}y_i)\,d^3y,
\qquad \ddot I_{ij}=2\int T^{ij}\,d^3y.
\tag{10}
\]

The monopole is constant and the mass dipole vanishes in the center-of-mass frame. For this rigid rotor $I_{aa}$ is also constant. Its oscillating second moment is therefore STF. The stationary angular-momentum part of $\int T^{0i}y_j$ does not contribute at $\Omega$.

Let

\[
F_{ij}(t,\mathbf x)=\frac{Q_{ij}(t-r/c)}r.
\tag{11}
\]

The three expressions above therefore give

\[
\bar h^{00}_Q=\frac{2G}{c^2}\partial_{ab}F_{ab},\qquad
\bar h^{0i}_Q=-\frac{2G}{c^3}\partial_a\dot F_{ia},\qquad
\bar h^{ij}_Q=\frac{2G}{c^4}\ddot F_{ij}.
\tag{12}
\]

For example, $\partial_0\bar h^{00}+\partial_i\bar h^{i0}=0$ follows immediately from commuting derivatives; the spatial harmonic constraints cancel in the same way. Each field satisfies the exterior wave equation. This conserved package is valid for arbitrary $kr$ at the retained source-multipole order.

This is the linear canonical mass-quadrupole sector of the standard exterior multipolar solution. The sign must be translated when comparing with literature using the gothic inverse metric $\sqrt{-g}g^{\mu\nu}-\eta^{\mu\nu}$, which equals $-\bar h^{\mu\nu}$ at first order. See the linear multipolar field in [Blanchet, *Quadrupole-quadrupole gravitational waves*, Section 2](https://arxiv.org/pdf/gr-qc/9710037).

## 4. Recover the physical metric and expose the missing terms

Trace reversal and index lowering are necessary:

\[
h_{\mu\nu}=\bar h_{\mu\nu}-\tfrac12\eta_{\mu\nu}\bar h.
\tag{13}
\]

For the oscillating STF quadrupole, $\bar h^{aa}=0$, so

\[
h_{00}^Q=\frac{G}{c^2}\partial_{ab}F_{ab},\quad
h_{0i}^Q=\frac{2G}{c^3}\partial_a\dot F_{ia},\quad
h_{ij}^Q=\delta_{ij}h_{00}^Q+\frac{2G}{c^4}\ddot F_{ij}.
\tag{14}
\]

Use $n_i=x_i/r$, $u=t-r/c$, $\partial_i r=n_i$, $\partial_i u=-n_i/c$, and $\partial_i n_j=(\delta_{ij}-n_in_j)/r$. For any time function $A$,

\[
\partial_i\frac{A(u)}r=-n_i\left[\frac{A(u)}{r^2}+\frac{\dot A(u)}{cr}\right],
\tag{15}
\]

\[
\partial_{ij}\frac{A(u)}r=
n_in_j\left[\frac{3A}{r^3}+\frac{3\dot A}{cr^2}+\frac{\ddot A}{c^2r}\right]
-\delta_{ij}\left[\frac{A}{r^3}+\frac{\dot A}{cr^2}\right].
\tag{16}
\]

Consequently the physical components are

\[
h_{00}^Q=\frac{G}{c^2}n_an_b
\left[\frac{3Q_{ab}(u)}{r^3}+\frac{3\dot Q_{ab}(u)}{cr^2}
+\frac{\ddot Q_{ab}(u)}{c^2r}\right],
\tag{17}
\]

\[
h_{0i}^Q=-\frac{2G}{c^3}n_a
\left[\frac{\dot Q_{ia}(u)}{r^2}+\frac{\ddot Q_{ia}(u)}{cr}\right],\qquad
h_{ij}^Q=\delta_{ij}h_{00}^Q+\frac{2G}{c^4r}\ddot Q_{ij}(u).
\tag{18}
\]

The manuscript retains only the last term in $h_{ij}^Q$, then projects it. It drops the scalar spatial part, $h_{00}$, $h_{0i}$, and their effect on the test masses.

The static monopole must be added when a total metric is wanted:

\[
h_{00}^{M}=\frac{2GM}{c^2r},\qquad h_{ij}^{M}=\delta_{ij}\frac{2GM}{c^2r}.
\tag{19}
\]

There are also stationary quadrupole and spin fields. They set a background equilibrium and do not supply a 600 Hz spectral line for an ideal steady rotor. At 6 km, the monopole alone has $h_{00}^{M}=1.565\times10^{-26}$ and acceleration magnitude $1.172\times10^{-13}\ {\rm m\,s^{-2}}$. Neither is the amplitude to insert into a 600 Hz SNR calculation. Modulation of the source position, material distribution, or supports could turn nominally static moments into additional signals.

## 5. The physical field to compare is the tidal curvature

Metric components depend on coordinates. Define the electric tidal tensor

\[
E_{ij}=c^2R_{0i0j},\qquad \ddot\xi_i=-E_{ij}\xi_j
\tag{20}
\]

for infinitesimally separated freely falling bodies. The linear curvature is gauge invariant about Minkowski space. With our signs,

\[
E_{ij}=\tfrac12\left[c\partial_t\partial_i h_{0j}
+c\partial_t\partial_j h_{0i}-c^2\partial_{ij}h_{00}-\partial_t^2h_{ij}\right].
\tag{21}
\]

Substituting (14),

\[
E_{ij}=-\frac G2\partial_{ijab}F_{ab}
+\frac G{c^2}\left(\partial_{ia}\ddot F_{ja}+\partial_{ja}\ddot F_{ia}
-\frac{\delta_{ij}}2\partial_{ab}\ddot F_{ab}\right)
-\frac G{c^4}F_{ij}^{(4)}.
\tag{22}
\]

Thus the quadrupolar curvature contains terms scaling as

\[
G\left\{\frac{Q}{r^5},\frac{\dot Q}{cr^4},\frac{\ddot Q}{c^2r^3},
\frac{Q^{(3)}}{c^3r^2},\frac{Q^{(4)}}{c^4r}\right\}.
\tag{23}
\]

For explicit numerical evaluation, set $z=kr$, $q_i=\mathsf Q_{ij}n_j$, $S=n_i\mathsf Q_{ij}n_j$. Equation (22) becomes

\[
\mathsf E_{ij}=\frac{Ge^{iz}}{r^5}
\left[A\mathsf Q_{ij}+B(n_iq_j+n_jq_i)+C\delta_{ij}S+D n_in_jS\right],
\tag{24}
\]

\[
\begin{aligned}
A&=-3+3iz+3z^2-2iz^3-z^4,\\
B&=15-15iz-9z^2+4iz^3+z^4,\\
C&=(15-15iz-3z^2-2iz^3-z^4)/2,\\
D&=(-105+105iz+45z^2-10iz^3-z^4)/2.
\end{aligned}
\tag{25}
\]

Useful checks follow directly:

* $E_{ii}=0$ outside the source, as required in vacuum.
* For $kr\to0$, $E_{ij}\to-\partial_{ij}U_Q$, with $U_Q=3GQ_{ab}n_an_b/(2r^3)$.
* For $kr\gg1$, $\mathsf E_{ij}\to-Gk^4e^{ikr}\mathsf Q_{ij}^{TT}/r$. This equals $\Omega^2\mathsf h_{ij}^{TT}/2$, the expected outgoing radiative result.

The approximate metric enhancement is $1/(kr)^2$, but the approximate tidal/free-mass strain enhancement over radiation is $1/(kr)^4$. At the vertex, $(kR)^{-2}=175.7$ and $(kR)^{-4}=3.086\times10^4$. Angular factors and the field variation across these long arms change the actual ratio; they explain why the numerical detector ratio is about $8.37\times10^4$, rather than either simple estimate. Near a response zero, such ratios have no universal meaning.

## 6. From curvature to the finite-arm observable

Take the vertex $B$ at zero, end mirrors at $L\mathbf e_x,L\mathbf e_y$, source center $\mathbf X=R\mathbf n_s$, and $T=L/c$. First model the optical reference and both end masses as free at the signal frequency, with negligible unperturbed velocities. The observable is the difference of round-trip proper light times at a common receiving vertex, normalized by $2L/c$. This is an ideal equal-arm Michelson; cavity calibration is addressed in Section 10.

### 6.1 Harmonic-gauge derivation: moving endpoints and light propagation

The geodesic equation at first order gives the coordinate acceleration

\[
\mathsf a_i=\frac{c^2}{2}\partial_i\mathsf h_{00}+ic\Omega\mathsf h_{0i}
=\frac G2\mathsf Q_{ab}\partial_{iab}g_k+2Gk^2\mathsf Q_{ia}\partial_ag_k.
\tag{26}
\]

For the periodic free-mass solution, $\mathsf\xi=-\mathsf a/\Omega^2$. Constant positions/velocities and DC gravitational displacements are part of the background, not this phasor.

For an arm in direction $\mathbf p$, its endpoint contribution to the round-trip length perturbation, received at time $t$, is

\[
\mathsf D_p^{\rm end}=p_i\left[2e^{i\Omega T}\mathsf\xi^i_{E_p}
-(1+e^{2i\Omega T})\mathsf\xi^i_B\right].
\tag{27}
\]

The mirror is sampled at the reflection time (t-T), and the vertex at both emission (t-2T) and reception $t$. Equation (27) follows by adding the changes in the two endpoint separations.

For a null ray, write $c\,dt=(1+\epsilon)ds$. Expanding $ds_{\rm spacetime}^2=0$ gives $\epsilon=(h_{00}+2p^ih_{0i}+p^ip^jh_{ij})/2$ on the outward path. Reversing the ray changes the sign of the mixed term. Therefore

\[
\begin{aligned}
\mathsf D_p^{\rm path}=\frac12\int_0^Lds\,\{&
[\mathsf h_{00}+2p^i\mathsf h_{0i}+p^ip^j\mathsf h_{ij}]
e^{ik(2L-s)}\\
&+[\mathsf h_{00}-2p^i\mathsf h_{0i}+p^ip^j\mathsf h_{ij}]
e^{iks}\},
\end{aligned}
\tag{28}
\]

where every metric is evaluated at $\mathbf r=s\mathbf p-\mathbf X$. Its $e^{ikr}$ factor already accounts for source-to-field propagation. The additional exponential factors describe the photon sampling times and must not be mistaken for a second source retardation.

The vertex proper-time conversion adds $-\tfrac12\int_{t-2T}^{t}h_{00}(t',B)dt'$ to each arm. It is identical for equal arms at the same receiving event and cancels in the differential signal. Then

\[
\mathsf H=\frac1{2L}
[(\mathsf D_x^{\rm end}+\mathsf D_x^{\rm path})
-(\mathsf D_y^{\rm end}+\mathsf D_y^{\rm path})].
\tag{29}
\]

Only the sum has invariant observational meaning. The separate endpoint and path pieces depend on gauge. For unequal arms, different clock arrangements, or driven endpoints, their appropriate clock and mechanical terms must be retained explicitly.

### 6.2 A compact equivalent formula using curvature

For a nonzero Fourier frequency we may transform to synchronous coordinates, $h_{00}^{s}=h_{0i}^{s}=0$, in which the periodic freely falling reference bodies have fixed spatial coordinates. With the convention $h'_{\mu\nu}=h_{\mu\nu}-\partial_\mu\zeta_\nu-\partial_\nu\zeta_\mu$, one choice is

\[
\mathsf\zeta_0=\frac{ic}{2\Omega}\mathsf h_{00},\qquad
\mathsf\zeta_i=\frac{ic}{\Omega}\mathsf h_{0i}
+\frac{c^2}{2\Omega^2}\partial_i\mathsf h_{00}.
\tag{30}
\]

Equation (21) then implies $\mathsf h^s_{ij}=2\mathsf E_{ij}/\Omega^2$. This synchronous field is not generally transverse. Its trace vanishes in the present vacuum sector because $E_{ii}=0$. No local TT projection has been used.

Substituting into the null-path integral gives the particularly convenient result

\[
\mathsf H=\frac1{2L\Omega^2}
\left[\int_0^L W(s)\mathsf E_{xx}(s\mathbf e_x-\mathbf X)\,ds
-\int_0^L W(s)\mathsf E_{yy}(s\mathbf e_y-\mathbf X)\,ds\right],
\tag{31}
\]

\[
W(s)=e^{ik(2L-s)}+e^{iks}=2e^{ikL}\cos[k(L-s)].
\tag{32}
\]

Equations (22) and (31) are the proposed replacement for the manuscript's field and ideal single-source response. They retain every radial term of the leading quadrupole and the full arm geometry. They also provide an independent check of (29). Curvature-based descriptions of optical detector response have a broader covariant foundation; see [Koop and Finn, *Gravitational wave detector response in terms of spacetime Riemann curvature*](https://arxiv.org/abs/1310.2871). Equation (31) here is derived specifically for the stated periodic free-body, equal-arm setup, rather than imported as a formula for a suspended cavity detector.

## 7. The dominant physical effect: differential gravitational acceleration

In the near-zone Newtonian limit define ($U>0$) for positive mass, so $\mathbf g=\nabla U$. The quadrupole potential and acceleration are

\[
U_Q=\frac{3G}{2r^3}Q_{ij}n_in_j,
\qquad
\mathsf{\mathbf g}_Q=\frac{G}{r^4}
\left[3\mathsf Q\mathbf n-\frac{15}{2}(\mathbf n^T\mathsf Q\mathbf n)\mathbf n\right].
\tag{33}
\]

Since $E_{pp}=-\partial_p g_p$, integrating (31) with $W\simeq2$ gives






\[
\mathsf H_N=-\frac{[\mathsf g_x(L\mathbf e_x)-\mathsf g_x(0)]
-[\mathsf g_y(L\mathbf e_y)-\mathsf g_y(0)]}{\Omega^2L}.
\tag{34}
\]

This endpoint difference keeps the full spatial field variation; it is not the small-$L/R$ geodesic-deviation approximation. In the further limit $L/R\to0$, it reduces to $(\mathsf E_{xx}-\mathsf E_{yy})/\Omega^2$. A common uniform gravitational acceleration cancels in (34), as required.

Thus a single rotor mainly drives relative mirror motion. For a nearby single endpoint its displacement scales approximately $GQ/(\Omega^2 d^4)$, where $d$ is source-to-mirror separation. If the complete detector is distant compared with its arm length, subtracting the common acceleration instead gives $H_N\sim GQ/(\Omega^2R^5)$. The $d^{-4}$ and $R^{-5}$ statements concern different geometrical limits.

For fixed rotor shape and density, $Q$ is independent of spin speed. Newtonian force amplitude at the rotating pattern frequency is nearly frequency independent, while a free-mass displacement falls as $f_0^{-2}$. This differs from the radiative strain's $f_0^2$ scaling. Therefore frequency and arm-length optimization must be repeated with the new response and actual noise; increasing rotor speed or arm length does not automatically improve a near-zone interaction measurement.

Rotating gravitational calibrators have already driven interferometer test masses through this interaction. [Estevez et al., *First Tests of a Newtonian Calibrator on an Interferometric Gravitational Wave Detector*](https://arxiv.org/pdf/1806.06572), Section 2, explicitly connects rotor gravitational force to free-mirror displacement and tests its distance and phase dependence. [Ross et al., *Initial Results from the LIGO Newtonian Calibrator*](https://arxiv.org/abs/2107.00141) reports resolved quadrupole and hexapole gravitational forcing in LIGO. These establish the detection principle, not the engineering or sensitivity of the present kilometre-scale proposal.

## 8. Rotor moments and numerical results

### 8.1 The actual source used

The current `configs/source.yaml` specifies a uniform rotor with $D=5$ m, axial length $H=2$ m, density $1750\ {\rm kg\,m^{-3}}$, and two opposed cylindrical holes of diameter 1 m at radius $s=1.5$ m. The spin frequency is 300 Hz. Each missing mass has magnitude

\[
m_h=\rho\frac{\pi d^2H}{4}=2748.89357189\ {\rm kg},\quad
M=63224.5521535\ {\rm kg},\quad m_hs^2=6185.01053675\ {\rm kg\,m^2}.
\tag{35}
\]

The uniform cylinder's Eulerian density is time independent. Each round hole's intrinsic second moment is invariant under rotation about the rotor axis. Their only time-varying Newtonian quadrupole is therefore exactly the orbital second moment of the missing mass. At zero mechanical phase its body-frame peak phasor is

\[
\mathsf Q_{\rm body}=-m_hs^2
\begin{pmatrix}1&i&0\\i&-1&0\\0&0&0\end{pmatrix},\qquad
\mathsf Q=\mathcal R\mathsf Q_{\rm body}\mathcal R^T.
\tag{36}
\]

The rotation matrix is exactly the one in `ghe/geometry.py`; the body $x,y$ completion affects the mechanical phase convention.

### 8.2 Recomputed signals, rather than total metric norms

All entries below are peak amplitudes. “Full” means the complete leading mass-quadrupole field, including retardation, in the ideal free-mass equal-arm observable (31). “Newtonian” means (34). The third column is an illustrative placement, not a claim of a globally optimized apparatus.

| Quantity | Paper benchmark, cached angles | Former 3995 m defaults, cached angles | Paper benchmark, source $+y$, rotor axis $+z$ |
|---|---:|---:|---:|
| $L$, m | 4000 | 3995 | 4000 |
| Source-to-vertex $R$, m | 6000 | 5992.5 | 6000 |
| Nearest end-mirror distance, m | 5701.88 | 5694.75 | 2000 |
| Differential Newtonian acceleration, ${\rm m\,s^{-2}}$ | $1.10512\times10^{-21}$ | $1.11066\times10^{-21}$ | $1.15167\times10^{-19}$ |
| $\vert H_N\vert$ | $1.94396\times10^{-32}$ | $1.95615\times10^{-32}$ | $2.02585\times10^{-30}$ |
| $\vert H_{\rm full}\vert$ | $1.93585\times10^{-32}$ | $1.94801\times10^{-32}$ | $2.02610\times10^{-30}$ |
| Equivalent differential length $L\vert H_{\rm full}\vert$, m | $7.74339\times10^{-29}$ | $7.78231\times10^{-29}$ | $8.10440\times10^{-27}$ |
| Existing local-TT radiative template $\vert H_{\rm old}\vert$ | $2.31286\times10^{-37}$ | $2.31576\times10^{-37}$ | $4.74512\times10^{-38}$ |
| Full / old amplitude | $8.36992\times10^4$ | $8.41198\times10^4$ | $4.26987\times10^7$ |
| Full / Newtonian amplitude minus one | $-0.417225\%$ | $-0.416184\%$ | $+0.012374\%$ |

In the last column the angles are $(\pi/2,\pi/2,0,0)$. The end mirror is much closer to the source, which strengthens its Newtonian acceleration. This illustrates why the old radiative optimization should not be retained. It is not a guarantee that a real site permits this placement.

At the paper benchmark the complex quantities are

\[
\mathsf H_N=-1.93912543853\times10^{-32}-i\,1.36984874606\times10^{-33},
\tag{38}
\]

\[
\mathsf H_{\rm full}=-1.92173135524\times10^{-32}-i\,2.33351791085\times10^{-33}.
\tag{39}
\]

Their phase difference is $0.0503109$ rad, almost entirely the common photon delay $kL=0.0502990$ rad, rather than a source phase delay $kR=0.0754485$ rad.

In harmonic coordinates, the endpoint and path amplitudes for this case are $1.93625\times10^{-32}$ and $4.01095\times10^{-36}$. They must be added as complex values, not positive amplitudes. Their separation is gauge dependent; in synchronous coordinates the same total appears entirely in the light-path integral. It would be incorrect to identify the harmonic path term alone with a separately measured gravitational wave.

The static monopole dominates the total metric norm. The oscillating physical $h_{00}^Q$ at the vertex in the cached geometry is only $3.84531\times10^{-36}$, owing partly to its orientation. This small vertex component still produces a much larger effective mirror strain through spatial derivatives and the $1/\Omega^2$ mechanical response. Neither a single component nor a full tensor norm substitutes for (31).

### 8.3 Check higher source multipoles instead of assuming them away

The perfectly opposed, symmetric holes give inversion symmetry. Odd mass multipoles, including the mass octupole, vanish. The current dipole is stationary, the current quadrupole vanishes by parity, and the leading varying current multipole is a current octupole. Its contribution to the near-zone electric tidal field is suppressed parametrically by $(v/c)(ka)\sim O(v^2/c^2)$, around $10^{-10}$ for this source. Slow-motion corrections to the mass quadrupole are similarly small. These estimates assume the ideal symmetry and no additional moving support masses.

The next allowed mass multipole is $\ell=4$. Its near-zone contribution is generically of relative order $(a/r_{\min})^2$, with geometry-dependent coefficients and possible exceptions near a quadrupole response null. The entire arm stays outside the source; even in the arm-extension example $r_{\min}=2$ km, so the expansion remains well controlled.

To verify the actual correction, `scr/verifyNearFieldNewtonian.py` integrates the unexpanded instantaneous force over both negative-density cylindrical holes. The stationary full cylinder has exactly zero 600 Hz Fourier component. It computes the forces at the three reference points, subtracts the DC force with 50-digit arithmetic, extracts the $2\omega_{\rm rot}$ phasor, and forms (34). Geometry and quadrature inputs remain double precision; this is a cancellation-resistant force check, not a claim of 50-digit physical accuracy.

| Geometry, paper benchmark | Direct point-hole amplitude correction to quadrupole | Direct finite-cylinder amplitude correction to quadrupole |
|---|---:|---:|
| Cached angles | $-7.51052\times10^{-8}$ | $-2.08626\times10^{-8}$ |
| Source $+y$, rotor axis $+z$ | $+3.93313\times10^{-7}$ | $+1.09253\times10^{-7}$ |

Changing the cylinder quadrature from $(3,8,3)$ radial/azimuthal/axial points and 32 samples per spin to $(4,12,4)$ and 64 samples changes the reported strain by at most $3.5\times10^{-16}$ relatively in the four checked cases. The small direct-versus-quadrupole difference is therefore resolved numerically. These finite-size corrections are far below the omitted near-zone quadrupole effect. The check validates the Newtonian higher-multipole remainder at these geometries, not every relativistic current or support contribution.

## 9. Coherent phase alignment still works

Let $\mathsf K_a(\Omega)$ be source $a$'s complete response at zero mechanical phase, evaluated with its own position, rotor axis, detector response, and calibration. Every term arising from that source's $2\omega_{\rm rot}$ harmonic shares the same mechanical phase factor, regardless of its radial power. If its physical rotor phase is $\omega_{\rm rot}t+\alpha_a$, then in our $e^{-i\Omega t}$ convention

\[
\mathsf H_{\rm total}=\sum_a\mathsf K_a e^{-2i\alpha_a},\qquad
\alpha_a=\frac{\arg\mathsf K_a-\psi_{\rm target}}2\pmod\pi.
\tag{40}
\]

This yields $\vert H_{\rm total}\vert =\sum_a\vert K_a\vert$ for perfect alignment. With similar amplitudes it gives amplitude and SNR proportional to $N$, and signal power or SNR squared proportional to $N^2$ if detector noise stays fixed. SNR itself does not scale as $N^2$.

There is no need to align $1/r^3$, $1/r^2$, and $1/r$ contributions separately: they are parts of one complex transfer at one frequency. A phase shift multiplies the whole response. Alignment at one detector channel is not necessarily alignment everywhere in space or in another channel.

The old offsets must nevertheless be recomputed. Near-zone phasors do not have a simple phase $kR$. For example, the scalar quadrupole kernel is

\[
\mathsf h_{00}^Q=\frac{G}{c^2r^3}e^{iz}(3-3iz-z^2)\,n_an_b\mathsf Q_{ab}.
\tag{41}
\]

Expanding this full product, not just $e^{iz}$, cancels the first-order retardation phase. In the curvature kernel (24), the imaginary terms through cubic order cancel; the first outgoing imaginary term is

\[
\mathsf E_{ij}^{\rm imag,leading}=-\frac{2iGk^5}{5}\mathsf Q_{ij}.
\tag{42}
\]

For a real quadrupole component with a nonzero Newtonian response, intrinsic outgoing-field phase effects thus begin at relative order $(kr)^5$, while amplitude changes start at $(kr)^2$. A rotating quadrupole is complex: changes of the real angular transfer can also change the measured phase at lower order, especially near cancellations. The full response includes the optical factor (32). Hence “negligible propagation lag” is not a substitute for computing $\arg K_a$, and this cancellation is not evidence of instantaneous gravity or a way to infer a superluminal propagation speed.

The existing phase code uses $e^{+i\Omega t}$, so its phasor is the complex conjugate of the one in this report. It also implements its stored rotor offset as a time delay $t-\alpha_{\rm stored}/\omega_{\rm rot}$, corresponding to physical phase $-\alpha_{\rm stored}$. Preserve these signs explicitly when integrating the new kernel. For our convention a pure sinusoid is extracted as $K_-=h(0)+i h(\pi/(2\Omega))$; the existing convention uses the opposite imaginary sign.

For small independent Gaussian mechanical phase errors of rms $\sigma_\alpha$, the mean coherent amplitude is reduced by $e^{-2\sigma_\alpha^2}$. For example, $\sigma_\alpha\lesssim0.071$ rad (about 4 degrees) keeps this mean loss below 1%. A long observation also needs a tracked phase history: an untracked frequency error accumulates as $2\pi\delta f\,T_{\rm obs}$. Rotor encoders and a shared time reference can supply the template for demodulation; a short simulation does not establish year-long coherence.

Higher spatial multipoles can produce additional temporal harmonics. A single rotor phase shifts its $m\omega_{\rm rot}$ harmonic by $m\alpha$; it generally cannot independently align all harmonics and all detector channels. For a selected 600 Hz channel use the complete Fourier coefficient there. For a multi-harmonic search calculate and match each harmonic, rather than recovering a whole waveform from two samples.

Finally, for a spatially extended array $K_a$ varies strongly with distance and geometry in the near zone. Sum actual source positions and responses. Do not replace a large array by a single quadrupole at its center unless its entire extent is small compared with all detector separations and the relevant wavelength. A lattice's increasingly distant sources, geometric sign changes, finite clearances, and source-induced noise invalidate automatic extrapolation of a single value by $N$.

## 10. How to revise the SNR estimation

### 10.1 Keep the correct Fourier normalization

The repository's basic one-sided-PSD normalization is already correct. For a physical output $y$,

\[
\rho^2=4\int_0^\infty\frac{|\tilde y_{\rm sig}(f)|^2}{S_y(f)}\,df.
\tag{43}
\]

For $y(t)=\Re[\mathsf Y e^{-i2\pi f_0t}]$, a long coherent observation, and locally smooth one-sided noise,

\[
\rho^2\simeq\frac{|\mathsf Y|^2T_{\rm obs}}{S_y(f_0)}.
\tag{44}
\]

Here $\mathsf Y$ is a peak phasor. There is no extra factor of $\sqrt2$. If using an rms amplitude instead, $\vert Y\vert ^2=2Y_{\rm rms}^2$. `ghe/spectrum.py` uses $\Delta t\,\mathrm{rfft}(h)$; `ghe/snr.py` implements (43) and the monochromatic specialization. These operations can be retained.

### 10.2 Put the new signal and noise in the same calibrated channel

Let $R_h(f)$ be the calibration transfer from a reference gravitational-wave strain to output $y$, with $S_h=S_y/\vert R_h\vert ^2$. Define

\[
\mathsf h_{\rm eff}(f)=\frac{\mathsf y_{\rm gravity}(f)}{R_h(f)},\qquad
\rho^2=\frac{|\mathsf h_{\rm eff}(f_0)|^2T_{\rm obs}}{S_h(f_0)}.
\tag{45}
\]

This definition does not imply that the source is a radiation-zone GW. It expresses any modeled gravitational signal in the detector's calibrated strain units.

For the leading force channel, first calculate the accelerations of the actual optical test masses. In a Fabry–Perot Michelson, use

\[
\delta L_{\rm DARM}=\mathbf e_x\cdot(\boldsymbol\xi_{\rm ETMx}-\boldsymbol\xi_{\rm ITMx})
-\mathbf e_y\cdot(\boldsymbol\xi_{\rm ETMy}-\boldsymbol\xi_{\rm ITMy}),
\tag{46}
\]

with the actual input-mirror positions and mechanical responses. The three-point model approximates both input references by a common freely moving vertex. Their actual positions, beam offsets, mirror rotations, support motions, and control loops are not specified by the current source geometry. Those details matter for a precision apparatus forecast.

For a simple suspended mass,

\[
\boldsymbol\xi_A=\chi_A(\Omega)m_A\mathbf g_A+\hbox{support/control contributions},\quad
\chi_A=\frac1{m_A(\omega_A^2-\Omega^2-i\Gamma_A\Omega)}.
\tag{47}
\]

At frequencies well above the relevant mechanical resonances and away from internal modes, $\chi_A m_A\simeq-1/\Omega^2$; gravity-driven displacement is then independent of mirror mass. Mirror mass still changes quantum noise and feedback dynamics. A detuned optical spring requires the coupled mechanical/optical transfer, rather than the bare oscillator formula alone.

There is a useful reason the free-mass equivalent strain can often be compared with calibrated DARM strain noise. If both the gravitational force signal and the reference long-wavelength GW drive the same DARM acceleration coordinate, write $y=R_x\chi_a a_{\rm DARM}$. The reference wave corresponds to $a_{\rm GW}=-\Omega^2Lh$. The common susceptibility and readout cancel in their ratio:

\[
h_{\rm eff}^{\rm force}=\frac{a_{\rm DARM}}{-\Omega^2L}.
\tag{48}
\]

This justifies (34) as a leading calibrated-force estimate under that equivalence, even when a common response is present. It does not justify multiplying it by an additional optical-spring or signal-recycling “gain” while leaving a strain-referred noise curve unchanged. Distributed gravitational light-path effects must be propagated through the same cavity model, including repeated trips and calibration. Equation (31) is an exact single-roundtrip free-body result, not by itself that cavity transfer.

The present full-versus-Newtonian comparison and $kL\simeq0.05$ make (45) with the ideal response a useful first estimate. The quoted SNRs use this explicitly labeled proxy. Their numerical precision is not a measure of the uncertainty in an actual detector model.

### 10.3 Recomputed conditional SNRs and noise limitations

Using the current ideal detuned model, 10 dB squeezing, the source frequency 600 Hz, and $T_{\rm yr}=365\times24\times3600$ s, the recomputed values are:

| Quantity | Paper $L,R$, cached angles | Former 3995 m defaults, cached angles | Paper $L,R$, arm-extension example |
|---|---:|---:|---:|
| Ideal quantum ASD, ${\rm Hz}^{-1/2}$ | $9.30653\times10^{-27}$ | $9.30659\times10^{-27}$ | $9.30653\times10^{-27}$ |
| One-source full-response SNR/year | 0.0116812 | 0.0117545 | 1.22258 |
| Ideal identical-response $N$ for SNR 5 | 428.04 | 425.37 | 4.09 |

The last row is an algebraic benchmark, not an array design. Integer counts would be at least 429, 426, and 5 under identical responses, fixed noise, and perfect coherence. Real placement and correlated disturbances must be evaluated source by source.

The former detector configuration had $L=3995$ m, test mass 200 kg, and $T_{\rm SRM}=10^{-5}$. The paper's appendix still quotes a 4000 m/6000 m benchmark and an older SNR value, while its parameter table has partly changed. The first column above changes $L,R$ to the requested benchmark but retains that former optical and source setup; it is not a reconstruction of every historical detector setting.

`ghe/noise.py` supplies a quantum-noise model with ideal frequency-dependent squeezing. It does not add a complete thermal, gas, seismic, control, or source-induced noise budget. As historical context, a separate GWINC 0.6.2 evaluation of `configs/aLIGO.yaml` at 600 Hz gave a total reference ASD $3.95376\times10^{-24}\ {\rm Hz}^{-1/2}$, about 425 times larger than the former ideal quantum proxy. That reference has a 39.6 kg test mass and different optics. It is not the CE2 hybrid detector's noise curve. With that reference ASD as a separate sensitivity illustration, the same two paper-geometry signal amplitudes would give only about $2.75\times10^{-5}$ and $2.88\times10^{-3}$ SNR/year.

For the experiment itself, construct the measured/calculated total output spectrum, including cross spectra for correlated terms. Schematically, for independent contributions only,

\[
S_y=S_y^{\rm quantum}+S_y^{\rm thermal}+S_y^{\rm seismic}+S_y^{\rm gas}
+S_y^{\rm source\,noise}+S_y^{\rm other}.
\tag{49}
\]

Stable rotor-synchronous vibration, acoustic, magnetic, electrical, or scattered-light lines are not automatically stationary random noise that can be combined by this sum. They can mimic the template and bias a gravitational amplitude measurement. They require a nuisance-response model and independent discrimination.

### 10.4 A practical replacement workflow

1. Load and record effective source, detector, geometry, and observation parameters. Pass each source position explicitly; repair the legacy distance reset before trusting array runs.
2. Construct the source's $Q_{ij}(t)$ or Fourier moments. Include higher harmonics or multipoles where a validated error target requires them. Include moving supports when their moments are relevant.
3. Evaluate (22)/(24) and (31) for an ideal baseline; use (26)–(29) with actual mechanics and optical transfer for the instrument model. Keep the result complex. Recover (34) as an independent low-frequency check.
4. Express the result as $y_{\rm gravity}$ or calibrated $h_{\rm eff}$. Optimize its SNR over geometry and frequency. Do not reuse radiative-only optimum angles.
5. Recompute source phase offsets from the new complete transfer. Sum phasors source by source, or sum resolved harmonics with the measured rotor phase histories.
6. Evaluate (43) or (45) against noise in the same channel. Test finite-source convergence, phase errors, array extent, and systematic discrimination before quoting a detection forecast.

For a monochromatic ideal diagnostic, `ghe.snr.calculate_snr_from_phasor` can accept the new peak effective phasor once the calibration assumption is explicitly selected. Its normalization need not change. Cached spectra, stored amplitudes, optimized angles, phase offsets, and array SNR tables do need regeneration after the signal model changes.

## 11. What can this apparatus establish?

If the experimental objective is detecting rotor-powered gravitational interaction, the leading Newtonian term is the desired signal. Model it and measure its amplitude, frequency, spatial dependence, and phase relation to rotor encoders. There is no reason to subtract it merely because it is not the radiative piece.

The experiment still needs to distinguish gravity from rotor-synchronous non-gravitational coupling. Useful tests include deliberately changing source position and axis, changing the quadrupole while controlling mechanical drive conditions, comparing predicted sign reversals, varying spin frequency, using gravitational-response null geometries, and recording local vibration/magnetic/acoustic witnesses. The source-to-mirror gravity law and spatial tensor pattern provide more discriminating information than one phase-locked tone. Rotational torques and beam offsets can also convert gravity into apparent length; see the [LIGO Newtonian-calibrator torque analysis](https://dcc.ligo.org/LIGO-T2100088/public).

The repository does not specify ground-motion transfer functions, residual rotor imbalance, magnetic moments, acoustic propagation, support modes, beam offsets, or relevant measured spectra. Consequently this analysis can quantify the non-radiative gravitational contribution, but cannot honestly assign numerical amplitudes to those non-gravitational disturbances. At the cached paper geometry, the useful gravity target is only $1.105\times10^{-21}\ {\rm m\,s^{-2}}$ differential acceleration; the apparatus disturbance model must be assessed against that scale in the same readout.

If the objective instead remains isolating emitted gravitational radiation, a large near-zone response is not evidence of having done so. At a single monochromatic readout, the Newtonian and radiative contributions share a rotor phase reference and cannot generally be fitted as independent physical processes without additional spatial/frequency information. Merely subtracting a large predicted signal does not establish a radiation measurement.

The cached-geometry radiative template is about $1.2\times10^{-5}$ of the full response. Even this suggests very demanding source, transfer, and background control for any attempted subtraction, and the local-TT template is not itself an invariant finite-distance radiative observable. A radiation-specific proposal needs a demonstrable radiation-zone arrangement ($R\gg c/\Omega\approx79.5$ km at 600 Hz), or a carefully defined multi-channel observable that isolates the appropriate radiative curvature behavior with quantified near-field cancellation. Such a proposal requires a new sensitivity analysis. The near-zone calculation supports reframing the present design as a coherent dynamic-gravity experiment.

## 12. Required manuscript and implementation revisions

| Existing location | Required revision |
|---|---|
| `paper/main.tex`, “Near Field Analysis”, labels `eq:near-field-raw-metric` and `eq:near-field-dynamic-tt` | Replace the locally projected radiative tensor with the conserved metric (12)–(18), or present the gauge-invariant curvature (22). State the small-source and slow-motion errors separately from $kR$. |
| `eq:forward-delay`, `eq:return-delay`, `eq:near-field-detector-response` | Use the full metric plus endpoint/clock response, or the equal-arm free-body curvature formula (31), with its assumptions. |
| `eq:far-field-limit` | Require $kR\gg1$ as well as the appropriate detector-size assumptions for a radiation-zone TT interpretation. |
| Single-source optimization, phase alignment, array-SNR tables, arm-length/frequency sweeps | Recompute using the complete calibrated response. Distinguish SNR $\propto N$ from SNR squared $\propto N^2$ under fixed-noise ideal coherence. |
| Abstract, motivation, interpretation, and conclusions | If using the stronger signal quantified here, describe a measurement of driven gravitational interaction. Do not identify it as a demonstrated radiative GW measurement. |
| `ghe/metric.py`, `ghe/optimization.py`, `ghe/source_array/phase.py`, `ghe/signal.py` | Introduce/select a full-response kernel and regenerate dependent caches. The existing local-TT kernel may be retained only as a clearly labeled legacy/radiative comparison. |
| `ghe/near_field.py` | Correct the trace-reversal labeling; use element-dependent retardation if calling an integral fully retarded; enforce a conserved source including stresses. A vertex tensor alone is not an interferometer response. |
| `ghe/config.py:SourceConfig.__post_init__` and distance callers | Historical defect: explicit distances were overwritten, and both forms became 5992.5 m under the former 3995 m defaults. The production implementation has since corrected this. |
| Detector-noise discussion | Separate the speculative ideal quantum curve from a complete apparatus noise and calibration model. Synchronize the paper's mixed 3995/4000 m baselines. |

The new diagnostic passes explicit Cartesian source coordinates, so it avoids the distance-override defect. This task adds a reviewed diagnostic and this report; it does **not** silently switch the legacy production pipeline, change its configuration semantics, regenerate its old arrays, or rewrite `paper/main.tex`. The table identifies the further production/manuscript edits needed to adopt this analysis.

## 13. Reproducibility and verification

Added files:

* `ghe/finite_distance.py`: conserved leading mass-quadrupole physical metric, coordinate acceleration, gauge-invariant tidal curvature, and two equivalent ideal Michelson calculations.
* The former `scr/nearFieldResponse.py` diagnostic reproduced the paper benchmark, the then-current 3995 m configuration, an arm-extension example, and a fixed-arm distance sweep. Its saved historical results are in `docs/near-field-analysis-results.json`.
* `scr/verifyNearFieldNewtonian.py`: independent analytic quadrupole and direct point-hole/finite-cylinder Newtonian calculation. Writes `docs/near-field-newtonian-check.json`.
* `tests/test_finite_distance.py`: Newtonian, vacuum, harmonic-constraint, radiation-zone, finite-arm gauge-equivalence, short-arm, and rotor-Fourier checks.

Run from the repository root:

```bash
conda activate gravitational-Hertz-experiment
python scr/nearFieldResponse.py
python scr/verifyNearFieldNewtonian.py
python -m pytest tests/test_finite_distance.py tests/test_near_field.py tests/test_metric_smoke.py tests/test_snr.py tests/test_spectrum.py -q
```

The direct integral offers `--quick` for one quadrature setting and `--extended` for a third. Both scripts offer output-path overrides and use the saved geometry when present; if the ignored cache is absent, they use and identify the embedded angle snapshot in (37). Source and detector parameters are read from the current configuration, and recorded with the outputs. The paper benchmark deliberately uses the source/optical parameters available on the analysis date; future configuration changes will alter numerical results.


