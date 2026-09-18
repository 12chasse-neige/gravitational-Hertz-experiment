> **Historical derivation:** The radiative-only detector calculation below has
> been superseded in production by the conserved finite-distance response. See
> [the current derivation](near-field-analysis.md) and [review guide](production-integration-review.md).

# Gravitational Hertz Experiment

## The GW Radiation under TT gauge 

This script is used to calculate the metric tensor of the gravitational wave induced by a spot-like source, which means the distance from the detector to the source is much longer than the size of the source but may be akin to the wavelength, say
$$
R \gg d \\
R \lesssim \lambda
$$

We assume that the source is a rotating column with four holes inside, where the diameter of the column is $D \approx 5 \text{m}$ while the holes has diameter of about $d \approx 1 \text{m}$. The columns are made of carbon fiber.

According to the linearized theory, the metric tensor at the detecter will be proportional to the second-order derivative of the quardrupole tensor,
$$
h_{ij}^{TT} = \frac{2 G}{R c^{4}} \ddot{I}_{ij}^{TT} (t_{\text{rev}})
$$
where the quadrupole tensor is defined by
$$
I_{ij} (t) = \int_{V'} \rho \left(x'_{i} x'_{j} - \frac{1}{3} \delta_{ij} r'^{2}\right)  \, \text{d} \tau' 
$$

For the columns under this circumstance, the calculation of the quadrupole tensor could be simplified to the linear superposition of a column with the density $\rho$ and four columns of the density $- \rho$.

After getting the numerical value of $h_{ij}$, we'll use the projector to get the TT mode:
$$
\Lambda_{ij, kl} = P_{ik} P_{jl} - \frac{1}{2} P_{ij} P_{kl}
$$
where 
$$
P = \begin{pmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 0
\end{pmatrix}
$$
So $h_{ij}^{TT} = \Lambda_{ij, kl} h_{kl}$

The first part of the script will calculate the components of the metric tensor for a rotating carbon fiber column and output the result in the TT gauge.

## Single Source Near Field Analysis

The radiation may be dominated by multipole terms. So we need a careful analysis of the higher pole terms in the expansion.
\[
h_{\mu \nu} (t, \vec{x}) = 4 G \int d^{3} {x'} \frac{T_{\mu \nu} 
(t - \frac{|\vec{x} - \vec{x'}|}{c}, \vec{x'})}{|\vec{x} - \vec{x'}|}
\]
Expand this term, using the approximation condition $\Omega d \ll c$ to make the $t - \frac{|\vec{x} - \vec{x’}|}{c} \to t - \frac{r}{c}$ in the EMT, and expand the denominator, we can get the first 3 terms for the radiation

### Expand source size, retaining all propagation dependence

Choose the source center of mass as the origin for the following field derivation, and define $r=\vert \mathbf x\vert >a$. For a Fourier component use peak phasors:

$$
X(t)=\Re[\mathsf X e^{-i\Omega t}].
$$

Equation (5) then has the spatial kernel $g_k(\mathbf x)=e^{ikr}/r$. Taylor expansion in source coordinates gives

$$
\frac{e^{ik|\mathbf x-\mathbf y|}}{|\mathbf x-\mathbf y|}
=\sum_{\ell=0}^{\infty}\frac{(-1)^\ell}{\ell!}
y_{i_1}\cdots y_{i_\ell}\partial_{i_1}\cdots\partial_{i_\ell}g_k(\mathbf x),
$$

$$
\bar{\mathsf h}^{\mu\nu}=\frac{4G}{c^4}
\sum_{\ell=0}^{\infty}\frac{(-1)^\ell}{\ell!}\partial_Lg_k
\int \mathsf T^{\mu\nu}(\mathbf y)y_L\,d^3y.
$$

The spatial derivatives act on both $1/r$ and $e^{ikr}$. We have not replaced them by $-n_i\partial_t/c$, the replacement that would discard the near-zone terms. Traces of higher raw moments can be regrouped into STF moments with finite-source retardation corrections; at leading slow-motion order these corrections are controlled by $ka$.

To connect the three metric sectors, define the ordinary second mass moment

$$
I_{ij}=\frac1{c^2}\int T^{00}y_i y_j\,d^3y,
\qquad Q_{ij}=I_{ij}-\tfrac13\delta_{ij}I_{aa}.
$$

Integrating the conservation equations by parts, with localized-source boundary terms vanishing,

$$
\dot I_{ij}=\frac1c\int(T^{0i}y_j+T^{0j}y_i)\,d^3y,
\qquad \ddot I_{ij}=2\int T^{ij}\,d^3y.
$$

The monopole is constant and the mass dipole vanishes in the center-of-mass frame. For this rigid rotor $I_{aa}$ is also constant. Its oscillating second moment is therefore STF. The stationary angular-momentum part of $\int T^{0i}y_j$ does not contribute at $\Omega$.

Let

$$
F_{ij}(t,\mathbf x)=\frac{Q_{ij}(t-r/c)}r.
$$

The three expressions above therefore give

$$
\bar h^{00}_Q=\frac{2G}{c^2}\partial_{ab}F_{ab},\qquad
\bar h^{0i}_Q=-\frac{2G}{c^3}\partial_a\dot F_{ia},\qquad
\bar h^{ij}_Q=\frac{2G}{c^4}\ddot F_{ij}.
$$

For example, $\partial_0\bar h^{00}+\partial_i\bar h^{i0}=0$ follows immediately from commuting derivatives; the spatial harmonic constraints cancel in the same way. Each field satisfies the exterior wave equation. This conserved package is valid for arbitrary $kr$ at the retained source-multipole order.

This is the linear canonical mass-quadrupole sector of the standard exterior multipolar solution. The sign must be translated when comparing with literature using the gothic inverse metric $\sqrt{-g}g^{\mu\nu}-\eta^{\mu\nu}$, which equals $-\bar h^{\mu\nu}$ at first order. See the linear multipolar field in [Blanchet, *Quadrupole-quadrupole gravitational waves*, Section 2](https://arxiv.org/pdf/gr-qc/9710037).

### Recover the physical metric and expose the missing terms

Trace reversal and index lowering are necessary:

$$
h_{\mu\nu}=\bar h_{\mu\nu}-\tfrac12\eta_{\mu\nu}\bar h.
$$

For the oscillating STF quadrupole, $\bar h^{aa}=0$, so

$$
h_{00}^Q=\frac{G}{c^2}\partial_{ab}F_{ab},\quad
h_{0i}^Q=\frac{2G}{c^3}\partial_a\dot F_{ia},\quad
h_{ij}^Q=\delta_{ij}h_{00}^Q+\frac{2G}{c^4}\ddot F_{ij}.
$$

Use $n_i=x_i/r$, $u=t-r/c$, $\partial_i r=n_i$, $\partial_i u=-n_i/c$, and $\partial_i n_j=(\delta_{ij}-n_in_j)/r$. For any time function $A$,

$$
\partial_i\frac{A(u)}r=-n_i\left[\frac{A(u)}{r^2}+\frac{\dot A(u)}{cr}\right],
$$

$$
\partial_{ij}\frac{A(u)}r=
n_in_j\left[\frac{3A}{r^3}+\frac{3\dot A}{cr^2}+\frac{\ddot A}{c^2r}\right]
-\delta_{ij}\left[\frac{A}{r^3}+\frac{\dot A}{cr^2}\right].
$$

Consequently the physical components are

$$
h_{00}^Q=\frac{G}{c^2}n_an_b
\left[\frac{3Q_{ab}(u)}{r^3}+\frac{3\dot Q_{ab}(u)}{cr^2}
+\frac{\ddot Q_{ab}(u)}{c^2r}\right],
$$

$$
h_{0i}^Q=-\frac{2G}{c^3}n_a
\left[\frac{\dot Q_{ia}(u)}{r^2}+\frac{\ddot Q_{ia}(u)}{cr}\right],\qquad
h_{ij}^Q=\delta_{ij}h_{00}^Q+\frac{2G}{c^4r}\ddot Q_{ij}(u).
$$

The manuscript retains only the last term in $h_{ij}^Q$, then projects it. It drops the scalar spatial part, $h_{00}$, $h_{0i}$, and their effect on the test masses.

The static monopole must be added when a total metric is wanted:

$$
h_{00}^{M}=\frac{2GM}{c^2r},\qquad h_{ij}^{M}=\delta_{ij}\frac{2GM}{c^2r}.
$$

There are also stationary quadrupole and spin fields. They set a background equilibrium and do not supply a 600 Hz spectral line for an ideal steady rotor. At 6 km, the monopole alone has $h_{00}^{M}=1.565\times10^{-26}$ and acceleration magnitude $1.172\times10^{-13}\ {\rm m\,s^{-2}}$. Neither is the amplitude to insert into a 600 Hz SNR calculation. Modulation of the source position, material distribution, or supports could turn nominally static moments into additional signals.

### The physical field to compare is the tidal curvature

Metric components depend on coordinates. Define the electric tidal tensor

$$
E_{ij}=c^2R_{0i0j},\qquad \ddot\xi_i=-E_{ij}\xi_j
$$

for infinitesimally separated freely falling bodies. The linear curvature is gauge invariant about Minkowski space. With our signs,

$$
E_{ij}=\tfrac12\left[c\partial_t\partial_i h_{0j}
+c\partial_t\partial_j h_{0i}-c^2\partial_{ij}h_{00}-\partial_t^2h_{ij}\right].
$$

Substituting (14),

$$
E_{ij}=-\frac G2\partial_{ijab}F_{ab}
+\frac G{c^2}\left(\partial_{ia}\ddot F_{ja}+\partial_{ja}\ddot F_{ia}
-\frac{\delta_{ij}}2\partial_{ab}\ddot F_{ab}\right)
-\frac G{c^4}F_{ij}^{(4)}.
$$

Thus the quadrupolar curvature contains terms scaling as

$$
G\left\{\frac{Q}{r^5},\frac{\dot Q}{cr^4},\frac{\ddot Q}{c^2r^3},
\frac{Q^{(3)}}{c^3r^2},\frac{Q^{(4)}}{c^4r}\right\}.
$$

For explicit numerical evaluation, set $z=kr$, $q_i=\mathsf Q_{ij}n_j$, $S=n_i\mathsf Q_{ij}n_j$. Equation (22) becomes

$$
\mathsf E_{ij}=\frac{Ge^{iz}}{r^5}
\left[A\mathsf Q_{ij}+B(n_iq_j+n_jq_i)+C\delta_{ij}S+D n_in_jS\right],
$$

$$
\begin{aligned}
A&=-3+3iz+3z^2-2iz^3-z^4,\\
B&=15-15iz-9z^2+4iz^3+z^4,\\
C&=(15-15iz-3z^2-2iz^3-z^4)/2,\\
D&=(-105+105iz+45z^2-10iz^3-z^4)/2.
\end{aligned}
$$

Useful checks follow directly:

* $E_{ii}=0$ outside the source, as required in vacuum.
* For $kr\to0$, $E_{ij}\to-\partial_{ij}U_Q$, with $U_Q=3GQ_{ab}n_an_b/(2r^3)$.
* For $kr\gg1$, $\mathsf E_{ij}\to-Gk^4e^{ikr}\mathsf Q_{ij}^{TT}/r$. This equals $\Omega^2\mathsf h_{ij}^{TT}/2$, the expected outgoing radiative result.

The approximate metric enhancement is $1/(kr)^2$, but the approximate tidal/free-mass strain enhancement over radiation is $1/(kr)^4$. At the vertex, $(kR)^{-2}=175.7$ and $(kR)^{-4}=3.086\times10^4$. Angular factors and the field variation across these long arms change the actual ratio; they explain why the numerical detector ratio is about $8.37\times10^4$, rather than either simple estimate. Near a response zero, such ratios have no universal meaning.

### From curvature to the finite-arm observable

Take the vertex $B$ at zero, end mirrors at $L\mathbf e_x,L\mathbf e_y$, source center $\mathbf X=R\mathbf n_s$, and $T=L/c$. First model the optical reference and both end masses as free at the signal frequency, with negligible unperturbed velocities. The observable is the difference of round-trip proper light times at a common receiving vertex, normalized by $2L/c$. This is an ideal equal-arm Michelson; cavity calibration is addressed in Section 10.

#### Harmonic-gauge derivation: moving endpoints and light propagation

The geodesic equation at first order gives the coordinate acceleration

$$
\mathsf a_i=\frac{c^2}{2}\partial_i\mathsf h_{00}+ic\Omega\mathsf h_{0i}
=\frac G2\mathsf Q_{ab}\partial_{iab}g_k+2Gk^2\mathsf Q_{ia}\partial_ag_k.
$$

For the periodic free-mass solution, $\mathsf\xi=-\mathsf a/\Omega^2$. Constant positions/velocities and DC gravitational displacements are part of the background, not this phasor.

For an arm in direction $\mathbf p$, its endpoint contribution to the round-trip length perturbation, received at time $t$, is

$$
\mathsf D_p^{\rm end}=p_i\left[2e^{i\Omega T}\mathsf\xi^i_{E_p}
-(1+e^{2i\Omega T})\mathsf\xi^i_B\right].
$$

The mirror is sampled at the reflection time $(t-T)$, and the vertex at both emission $(t-2T)$ and reception $t$. Equation (27) follows by adding the changes in the two endpoint separations.

For a null ray, write $c\,dt=(1+\epsilon)ds$. Expanding $ds_{\rm spacetime}^2=0$ gives $\epsilon=(h_{00}+2p^ih_{0i}+p^ip^jh_{ij})/2$ on the outward path. Reversing the ray changes the sign of the mixed term. Therefore

$$
\begin{aligned}
\mathsf D_p^{\rm path}=\frac12\int_0^Lds\,\{&
[\mathsf h_{00}+2p^i\mathsf h_{0i}+p^ip^j\mathsf h_{ij}]
e^{ik(2L-s)}\\
&+[\mathsf h_{00}-2p^i\mathsf h_{0i}+p^ip^j\mathsf h_{ij}]
e^{iks}\},
\end{aligned}
\tag{28}
$$

where every metric is evaluated at $\mathbf r=s\mathbf p-\mathbf X$. Its $e^{ikr}$ factor already accounts for source-to-field propagation. The additional exponential factors describe the photon sampling times and must not be mistaken for a second source retardation.

The vertex proper-time conversion adds $-\tfrac12\int_{t-2T}^{t}h_{00}(t',B)dt'$ to each arm. It is identical for equal arms at the same receiving event and cancels in the differential signal. Then

$$
\mathsf H=\frac1{2L}
[(\mathsf D_x^{\rm end}+\mathsf D_x^{\rm path})
-(\mathsf D_y^{\rm end}+\mathsf D_y^{\rm path})].
$$

Only the sum has invariant observational meaning. The separate endpoint and path pieces depend on gauge. For unequal arms, different clock arrangements, or driven endpoints, their appropriate clock and mechanical terms must be retained explicitly.

## Estimate the critical angular velocity for the source

The source column is made of carbon fiber with critical stress of about 2 - 5 GPa. In this part, we'll do a brief evaluation of the maximum angular velocity of the source.

For simplicity, we assume that the column is linear elastic and homogeneous isotropic with Young's modulus $E$ and Poisson's ratio $\nu$. While in equilibrium, the stress tensor will obey the formula
$$
\nabla \cdot \sigma + \rho \omega^{2} r \hat{e}_{r} = 0
$$

Expanding the nabla operator and the tensor in the cylinder coordinate, we'll get 
$$
\nabla = \hat{e}_{r} \frac{\partial}{\partial r} + \hat{e}_{\theta} \frac{\partial}{\partial \theta} + \hat{e}_{z} \frac{\partial}{\partial z}
$$

So 
$$
\nabla \cdot \sigma = \hat{e}_{r} \frac{\mathrm{d}}{\mathrm{d} r} \sigma_{rr} + \hat{e}_{r} \frac{\sigma_{rr} - \sigma_{\theta \theta}}{r}
$$
and the previous equation becomes 
$$
\frac{\mathrm{d}}{\mathrm{d} r} \sigma_{rr} + \frac{\sigma_{rr} - \sigma_{\theta \theta}}{r} + \rho \omega^{2} r = 0
$$

For isotropic material, the constitutive equations give out that
$$
\sigma_{rr} = \lambda (\epsilon_{rr} + \epsilon_{\theta \theta} + \epsilon_{zz}) + 2 \mu \epsilon_{rr} \\
\sigma_{\theta \theta} = \lambda (\epsilon_{rr} + \epsilon_{\theta \theta} + \epsilon_{zz}) + 2 \mu \epsilon_{\theta \theta}
$$
with $\lambda = \frac{E \nu}{(1 + \nu) (1 - 2 \nu)}$ and $\mu = \frac{E}{1 + \nu}$

Now we can solve the first-order ODE
$$
(\lambda + 2 \mu) \frac{\mathrm{d}^{2}}{\mathrm{d} r^{2}}{u}_{r} + \lambda \frac{\mathrm{d}}{\mathrm{d} r} \frac{u_{r}}{r} + \frac{2 \mu}{r} \left(\frac{\mathrm{d}}{\mathrm{d} r} u_{r} - \frac{u_{r}}{r}\right) + \rho \omega^{2} r = 0
$$
$$
(\lambda + 2 \mu) \frac{\mathrm{d}}{\mathrm{d} r}\left(\frac{1}{r} \frac{\mathrm{d}}{\mathrm{d} r} (r u_{r})\right) + \rho \omega^{2} r = 0
$$
with the boudary conditions
$$
\left. \sigma_{rr} = 0 \right|_{r = R}
$$

The solution to this equation is
$$
\sigma_{rr} (r) = \frac{\rho \omega^{2} (2 \lambda + 3 \mu)}{4 (\lambda + 2 \mu)} (R^{2} - r^{2}) \\
\sigma_{\theta \theta} (r) = \frac{\rho \omega^{2}}{4 (\lambda + 2 \mu)} ((2 \lambda + 3 \mu) R^{2} - (2\lambda + \mu) r^{2})
$$

So the maximum stress in the rotating column equals
$$
\sigma_{\text{max}} = \frac{\rho \omega^{2} (2 \lambda + 3 \mu)}{4 (\lambda + 2 \mu)} R^{2}
$$

Given the typical fators of carbon fiber[^1], we choose Youngs's modulus of about 50 GPa with Poisson ratio of about 0.27, where the critical angular velocity can be estimated by 
$$
\omega_{c} = \sqrt{\frac{4 \sigma_{c} (\lambda + 2 \mu)}{\rho (2 \lambda + 3 \mu) R^{2}}} \approx 673 \, \text{rad}/ \text{s}
$$

## Shall we cut the column to several small columns?

If we keep the total volume unchanged and cut one big column into $N^{3}$ small columns, the critical angular velocity will become
$$
\omega_{c}' = N \omega_{c0}
$$

So the $h_{ij}$ at the detector will become
$$
h_{ij}' = N^{2} \cdot \frac{1}{N^{5}} h_{ij} = \frac{1}{N^{3}} h_{ij}
$$

The total response, considering superposition of the $N^{3}$ little columns, will be
$$
N^{3} h_{ij}' = h_{ij}
$$
about the same order of magnitude compared to one big column.

## Calculate the SNR of Single Source to Decide How Many Sources We Need

### Detector Response for the GW Terms-only Case

First, we should get the signal to the detector while under our circumstance, the approximation $\lambda \gg L$ does not work. The time delay in the transition time of the photon is 
$$
\delta T(t) = \frac{1}{2c} a^{i} a^{j} \int_{0}^{L} h_{ij} \left(\text{Distance}(\xi), t_{0} + \frac{\xi}{c} - \frac{\text{Distance}(\xi)}{c} \right) \, \mathrm{d} \xi
$$
and
$$
\delta T'(t) = \frac{1}{2c} a^{i} a^{j} \int_{0}^{L} h_{ij} \left(\text{Distance}(\xi), t_{0} + \frac{L - \xi}{c} + \frac{\text{Distance}(\xi)}{c} \right) \, \mathrm{d} \xi
$$
where $t_{0} = t - T$.

So the perturbation of the whole round trip is
$$
\delta T_{r.t.} (t) = \delta T(t - T) + \delta T' (t)
$$
and we could get the signal input to the detector
$$
h(t) = \frac{\delta L}{L} = \frac{\delta T}{T} \\ = 
\frac{1}{4L} a^{i} a^{j} \int_{0}^{L} h_{ij} \left(\text{Distance}(\xi), t_{0} + \frac{\xi}{c} - \frac{\text{Distance}(\xi)}{c} \right) \\ + h_{ij} \left(\text{Distance}(\xi), t_{0} + \frac{L - \xi}{c} + \frac{\text{Distance}(\xi)}{c} \right) \, \mathrm{d} \xi
$$

### The SNR Calculation

With the signal input to the detector, we can now analysis the noise of the detector and get the SNR of a signal source. Assume that the whole signal we see is $d(t) = n(t) + h(t)$, so we can get the root mean square of the noise after filtering (whose mean value is assumed zero)
$$
\langle N^{2} (t) \rangle =  \int \mathrm{d} f_{1} \int \mathrm{d} f_{2} e^{i 2\pi (f_{1} - f_{2}) t} k(f_{1}) k^{*} (f_{2}) \langle \tilde{n}(f_{1}) \tilde{n}^{*} (f_{2}) \rangle
$$
By definition, we have
$$
\langle \tilde{n} (f_{1}) \tilde{n}^{*} (f_{2}) \rangle = \frac{1}{2} S_{n} (f_{1}) \delta (f_{1} - f_{2})
$$
So the former integral becomes
$$
\int_{- \infty}^{\infty} \mathrm{d} f \frac{S_{n} (f)}{2} |k(f)|^{2} = \int_{0}^{\infty} \mathrm{d} f S_{n} (f) |k(f)|^{2}
$$

Then we can calculate the SNR by doing the integral
$$
\text{SNR} = \frac{S}{\sqrt{\langle N^{2} \rangle}} = \sqrt{4\int_{0}^{\infty} \frac{|h (f)|^{2}}{S_{n}^{\text{(one-sided)}} (f)} \, \mathrm{d} f}
$$

## Design Low Quantum Noise Detector for the 600 Hz Wave

The quantum noise for the interferometer is (before squeezing)
$$
S^{\text{SQL}}_{h} = \frac{h_{SQL}^{2}}{2} \left(\frac{1}{\kappa} + \kappa\right)
$$
where the coupling constant is defined as
$$
h_{SQL} = \sqrt{\frac{8 \hbar}{m \Omega^{2} L^{2}}}, \quad
\kappa (\Omega) = \frac{2 \frac{I_{o}}{I_{SQL}} \gamma^{4}}{\Omega^{2} (\gamma^{2} + \Omega^{2})}
$$
and $I_{SQL}$ is the power when $\gamma = \Omega$, defined as
$$
I_{SQL} = \frac{m L^{2} \gamma^{4}}{4 \omega_{o}}
$$
Notice that the laser power $I_{o}$ is the power at the beam splitter.

### Magnifying at the PRM

The laser power circling in the cavities is different from the laser power input to the detector, mostly because the magnifying effect of the PRM mirror. First we consider the cavity on the arm (ITM and ETM), while steady, the power in and out the cavity at the ITM is
$$
G_{arm} = \frac{P_{in}}{P_{circ}} = \frac{T_{ITM}}{1 - \sqrt{R_{ITM} R_{ETM}}}
$$
Using the approximation $T_{ETM} \ll T_{ITM}$, the ratio is near
$$
\frac{P_{in}}{P_{circ}} \approx \frac{T_{ITM}}{4}
$$
Loss of per circulation is 
$$
\Lambda_{MI} = \frac{4 (T_{ETM} + L_{arm})}{T_{ITM}} + L_{BS}
$$
So the beam splitter and the arm F-P cavity can be regarded as one mirror with equivalent reflection ratio
$$
R_{MI} = 1 - \Lambda_{MI}
$$
Therefore, the gain for the equivalent cavity is
$$
G_{PR} = \frac{4 T_{PRM}}{(T_{PRM} + \Lambda_{MI})^{2}}
$$
Choosing $T_{PRM} = \Lambda_{MI}$, the total gain is approximately 
$$
G_{PR,max} \approx \frac{1}{\Lambda_{MI}}
$$

### Signal Recycling and Detuned Interferometers

Introduce detuned signal recycle can bring a minimum asd value, according to the noise spectrum given by[^2]Buonanno and Chen
\[
\begin{pmatrix} b_1 \\ b_2 \end{pmatrix} = \frac{1}{M} \left[ e^{2i(\beta + \Phi)} \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix} \begin{pmatrix} a_1 \\ a_2 \end{pmatrix} + \sqrt{2\mathcal{K}} \tau e^{i(\beta + \Phi)} \begin{pmatrix} D_1 \\ D_2 \end{pmatrix} \frac{h}{h_{\text{SQL}}} \right]
\]
where the transfer matrices are defined as
$$
M = 1 + \rho^2 e^{4i(\beta + \Phi)} - 2\rho e^{2i(\beta + \Phi)} \left( \cos 2\phi + \frac{\mathcal{K}}{2} \sin 2\phi \right)
$$

$$
C_{11} = C_{22} = (1 + \rho^2) \left( \cos 2\phi + \frac{\mathcal{K}}{2} \sin 2\phi \right) - 2\rho \cos (2(\beta + \Phi))
$$

$$
C_{12} = -\tau^2 (\sin 2\phi + \mathcal{K} \sin^2 \phi), \quad C_{21} = \tau^2 (\sin 2\phi - \mathcal{K} \cos^2 \phi)
$$

$$
D_1 = -(1 + \rho e^{2i(\beta + \Phi)}) \sin \phi, \quad D_2 = -(-1 + \rho e^{2i(\beta + \Phi)}) \cos \phi
$$

\[
\phi = \left. \frac{\omega_{o} l}{c} \right|_{\text{mod} \, 2 \pi}  \quad \Phi = \arctan \frac{\Omega}{\gamma}
\]

When we induce squeezing to  further reduce the quantum noise, we can get the psd curve for the squeezed noise[^3] 
\[
S_{h} = \frac{\begin{pmatrix}\cos \zeta & \sin \zeta \end{pmatrix} \mathbb{T} \mathcal{D} (- \lambda) \mathcal{S} (2r) \mathcal{D} (\lambda) \mathbb{T}^{\dagger} \begin{pmatrix} \cos \zeta \\ \sin \zeta \end{pmatrix}}{\begin{pmatrix}\cos \zeta & \sin \zeta \end{pmatrix} \bar{s} \bar{s}^{\dagger} \begin{pmatrix} \cos \zeta \\ \sin \zeta \end{pmatrix}}
\]
where
$$
T_{11,22} = \mathrm{e}^{2\imath\Phi} \left[ (1 + \rho^2) \left( \cos(2\phi) + \frac{K}{2} \sin(2\phi) \right) - 2\rho \cos(2\Phi) \right]
$$

$$
T_{12} = -\mathrm{e}^{2\imath\Phi} \tau^2 \left( \sin(2\phi) + K \sin^2(\phi) \right)
$$

$$
T_{21} = \mathrm{e}^{2\imath\Phi} \tau^2 \left( \sin(2\phi) - K \cos^2(\phi) \right)
$$

$$
M = 1 + \rho^2 \mathrm{e}^{4\imath\Phi} - 2\rho \mathrm{e}^{2\imath\Phi} \left( \cos(2\phi) + \frac{K}{2} \sin(2\phi) \right)
$$

\[
\phi = \left. \frac{\omega_{o} l}{c} \right|_{\text{mod} \, 2 \pi} \quad \Phi = \arctan \frac{\Omega}{\gamma}
\]

The signal transfer functions $\bar{s}$ for the two quadratures are given by
$$
\bar{s}_1 = -\frac{\sqrt{2K}}{h_{\mathrm{SQL}}} \tau \left( 1 + \rho \mathrm{e}^{2\imath\Phi} \right) \sin(\phi)
$$

$$
\bar{s}_2 = -\frac{\sqrt{2K}}{h_{\mathrm{SQL}}} \tau \left( -1 + \rho \mathrm{e}^{2\imath\Phi} \right) \cos(\phi)
$$

And the squeezing matrices are defined as
\[
\mathcal{D} (\lambda) = \begin{pmatrix}
\cos \lambda & \sin \lambda \\ - \sin \lambda & \cos \lambda
\end{pmatrix}, \quad 
\mathcal{S} (r) = \begin{pmatrix}
e^{r} & 0 \\
0 & e^{-r}
\end{pmatrix}
\]
Fixing $\zeta = \frac{\pi}{2}$, and using frequency-dependent squeezing to get the best noise (which means that the squeezing matrices are equivalent to an $e^{-r}$ factor), the psd curve in our project is
\[
S_{h} = e^{-2r} \frac{T_{21} T_{21}^{*} + T_{22} T_{22}^{*}}{s_{2} s_{2}^{*}}
\]

\[
S_{h} (\rho, \Omega) =
\frac{h^{2}_{SQL} e^{-2r}}{2 \mathcal{K}} \frac{(1 - \rho^{2})^{2} \left( \sin(2\phi) - \mathcal{K} \cos^2(\phi) \right)^{2} + \left[ (1 + \rho^2) \left( \cos(2\phi) + \frac{\mathcal{K}}{2} \sin(2\phi) \right) - 2\rho \cos(2\Phi) \right]^{2}}{ (1 - \rho^{2}) \cos^{2} \phi (1 - 2 \rho \cos (2 \Phi) + \rho^{2})}
\]

To make sure that the optical resonance is at the $600 \mathrm{Hz}$, we must adjust the $\phi$ to let the resonant 
peak be accurately the destination frequency.



[^1]: Sayed Abolfazl Mirdehghan, 1 - Fibrous polymeric composites, Editor(s): Masoud Latifi, In The Textile Institute Book Series, Engineered Polymeric Fibrous Materials, Woodhead Publishing, 2021, Pages 1-58, ISBN 9780128243817, https://doi.org/10.1016/B978-0-12-824381-7.00012-3. (https://www.sciencedirect.com/science/article/pii/B9780128243817000123)
[^2]: Buonanno, A., & Chen, Y. (2001). Quantum noise in second generation, signal-recycled laser interferometric gravitational-wave detectors. *Physical Review D*, *64*(4), 042006. https://doi.org/10.1103/PhysRevD.64.042006
[^3]: Harms, J., Chen, Y., Chelkowski, S., Franzen, A., Vahlbruch, H., Danzmann, K., & Schnabel, R. (2003). Squeezed-input, optical-spring, signal-recycled gravitational-wave detectors. *Physical Review D*, *68*(4), 042001. https://doi.org/10.1103/PhysRevD.68.042001
