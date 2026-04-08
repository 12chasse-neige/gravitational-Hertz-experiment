# Gravitational Hertz Experiment

## Calculate the metric tensor components in TT gauge 

This script is used to calculate the metric tensor of the gravitational wave induced by a spot-like source, which means the distance from the detector to the source is much longer than the size of the source but may be akin to the wavelength, say
\[
R \gg d \\
R \approx \lambda
\]

We assume that the source is a rotating column with four holes inside, where the diameter of the column is $D \approx 5 \text{m}$ while the holes has diameter of about $d \approx 1 \text{m}$. The columns are made of carbon fiber.

According to the linearized theory, the metric tensor at the detecter will be proportional to the second-order derivative of the quardrupole tensor,
\[
h_{ij}^{TT} = \frac{2 G}{R c^{4}} \ddot{I}_{ij}^{TT} (t_{\text{rev}})
\]
where the quadrupole tensor is defined by
\[
I_{ij} (t) = \int_{V'} \rho \left(x'_{i} x'_{j} - \frac{1}{3} \delta_{ij} r'^{2}\right)  \, \text{d} \tau' 
\]

For the columns under this circumstance, the calculation of the quadrupole tensor could be simplified to the linear superposition of a column with the density $\rho$ and four columns of the density $- \rho$.

After getting the numerical value of $h_{ij}$, we'll use the projector to get the TT mode:
\[
\Lambda_{ij, kl} = P_{ik} P_{jl} - \frac{1}{2} P_{ij} P_{kl}
\]
where 
\[
P = \begin{pmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 0
\end{pmatrix}
\]
So $h_{ij}^{TT} = \Lambda_{ij, kl} h_{kl}$

The first part of the script will calculate the components of the metric tensor for a rotating carbon fiber column and output the result in the TT gauge.

## Estimate the critical angular velocity for the source

The source column is made of carbon fiber with critical stress of about 2 - 5 GPa. In this part, we'll do a brief evaluation of the maximum angular velocity of the source.

For simplicity, we assume that the column is linear elastic and homogeneous isotropic with Young's modulus $E$ and Poisson's ratio $\nu$. While in equilibrium, the stress tensor will obey the formula
\[
\nabla \cdot \sigma + \rho \omega^{2} r \hat{e}_{r} = 0
\]

Expanding the nabla operator and the tensor in the cylinder coordinate, we'll get 
\[
\nabla = \hat{e}_{r} \frac{\partial}{\partial r} + \hat{e}_{\theta} \frac{\partial}{\partial \theta} + \hat{e}_{z} \frac{\partial}{\partial z}
\]

So 
\[
\nabla \cdot \sigma = \hat{e}_{r} \frac{\mathrm{d}}{\mathrm{d} r} \sigma_{rr} + \hat{e}_{r} \frac{\sigma_{rr} - \sigma_{\theta \theta}}{r}
\]
and the previous equation becomes 
\[
\frac{\mathrm{d}}{\mathrm{d} r} \sigma_{rr} + \frac{\sigma_{rr} - \sigma_{\theta \theta}}{r} + \rho \omega^{2} r = 0
\]

For isotropic material, the constitutive equations give out that
\[
\sigma_{rr} = \lambda (\epsilon_{rr} + \epsilon_{\theta \theta} + \epsilon_{zz}) + 2 \mu \epsilon_{rr} \\
\sigma_{\theta \theta} = \lambda (\epsilon_{rr} + \epsilon_{\theta \theta} + \epsilon_{zz}) + 2 \mu \epsilon_{\theta \theta}
\]
with $\lambda = \frac{E \nu}{(1 + \nu) (1 - 2 \nu)}$ and $\mu = \frac{E}{1 + \nu}$

Now we can solve the first-order ODE
\[
(\lambda + 2 \mu) \frac{\mathrm{d}^{2}}{\mathrm{d} r^{2}}{u}_{r} + \lambda \frac{\mathrm{d}}{\mathrm{d} r} \frac{u_{r}}{r} + \frac{2 \mu}{r} \left(\frac{\mathrm{d}}{\mathrm{d} r} u_{r} - \frac{u_{r}}{r}\right) + \rho \omega^{2} r = 0
\]
\[
(\lambda + 2 \mu) \frac{\mathrm{d}}{\mathrm{d} r}\left(\frac{1}{r} \frac{\mathrm{d}}{\mathrm{d} r} (r u_{r})\right) + \rho \omega^{2} r = 0
\]
with the boudary conditions
\[
\left. \sigma_{rr} = 0 \right|_{r = R}
\]

The solution to this equation is
\[
\sigma_{rr} (r) = \frac{\rho \omega^{2} (2 \lambda + 3 \mu)}{4 (\lambda + 2 \mu)} (R^{2} - r^{2}) \\
\sigma_{\theta \theta} (r) = \frac{\rho \omega^{2}}{4 (\lambda + 2 \mu)} ((2 \lambda + 3 \mu) R^{2} - (2\lambda + \mu) r^{2})
\]

So the maximum stress in the rotating column equals
\[
\sigma_{\text{max}} = \frac{\rho \omega^{2} (2 \lambda + 3 \mu)}{4 (\lambda + 2 \mu)} R^{2}
\]

Given the typical fators of carbon fiber, we choose Youngs's modulus of about 50 GPa with Poisson ratio of about 0.27, where the critical angular velocity can be estimated by 
\[
\omega_{c} = \sqrt{\frac{4 \sigma_{c} (\lambda + 2 \mu)}{\rho (2 \lambda + 3 \mu) R^{2}}} \approx 673 \, \text{rad}/ \text{s}
\]

## Shall we cut the column to several small columns?

If we keep the total volume unchanged and cut one big column into $N^{3}$ small columns, the critical angular velocity will become
\[
\omega_{c}' = N \omega_{c0}
\]

So the $h_{ij}$ at the detector will become
\[
h_{ij}' = N^{2} \cdot \frac{1}{N^{5}} h_{ij} = \frac{1}{N^{3}} h_{ij}
\]

The total response, considering superposition of the $N^{3}$ little columns, will be
\[
N^{3} h_{ij}' = h_{ij}
\]
about the same order of magnitude compared to one big column.

## Calculate the SNR of Single Source to Decide How Many Sources We Need

First, we should get the signal to the detector while under our circumstance, the approximation $\lambda \gg L$ does not work. The time delay in the transition time of the photon is 
\[
\delta T(t) = \frac{1}{2c} a^{i} a^{j} \int_{0}^{L} h_{ij} \left(\text{Distance}(\xi), t_{0} + \frac{\xi}{c} - \frac{\text{Distance}(\xi)}{c} \right) \, \mathrm{d} \xi
\]
and
\[
\delta T'(t) = \frac{1}{2c} a^{i} a^{j} \int_{0}^{L} h_{ij} \left(\text{Distance}(\xi), t_{0} + \frac{L - \xi}{c} + \frac{\text{Distance}(\xi)}{c} \right) \, \mathrm{d} \xi
\]
where $t_{0} = t - T$.

So the perturbation of the whole round trip is
\[
\delta T_{r.t.} (t) = \delta T(t - T) + \delta T' (t)
\]
and we could get the signal input to the detector
\[
h(t) = \frac{\delta T}{T} = \frac{1}{4L} a^{i} a^{j} \int_{0}^{L} h_{ij} \left(\text{Distance}(\xi), t_{0} + \frac{\xi}{c} - \frac{\text{Distance}(\xi)}{c} \right) + h_{ij} \left(\text{Distance}(\xi), t_{0} + \frac{L - \xi}{c} + \frac{\text{Distance}(\xi)}{c} \right) \, \mathrm{d} \xi
\]

With the signal input to the detector, we can now analysis the noise of the detector and get the SNR of a signal source. Assume that the whole signal we see is $d(t) = n(t) + h(t)$, so we can get the root mean square of the noise after filtering (whose mean value is assumed zero)
\[
\langle N^{2} (t) \rangle =  \int \mathrm{d} f_{1} \int \mathrm{d} f_{2} e^{i 2\pi (f_{1} - f_{2}) t} k(f_{1}) k^{*} (f_{2}) \langle \tilde{n}(f_{1}) \tilde{n}^{*} (f_{2}) \rangle 
\]
By definition, we have
\[
\langle \tilde{n} (f_{1}) \tilde{n}^{*} (f_{2}) \rangle = \frac{1}{2} S_{n} (f_{1}) \delta (f_{1} - f_{2})
\]
So the former integral becomes
\[
\int_{- \infty}^{\infty} \mathrm{d} f \frac{S_{n} (f)}{2} |k(f)|^{2} = \int_{0}^{\infty} \mathrm{d} f S_{n} (f) |k(f)|^{2}
\]

Then we can calculate the SNR by doing the integral
\[
\text{SNR} = \frac{S}{\sqrt{\langle N^{2} \rangle}} = \sqrt{4\int_{0}^{\infty} \frac{|h (f)|^{2}}{S_{n}^{\text{(one-sided)}} (f)} \, \mathrm{d} f}
\]

## Design Low Quantum Noise Detector for the 600 Hz Wave

The quantum noise for the interferometer is (before squeezing)
\[
S^{\text{SQL}}_{h} = \frac{h_{SQL}^{2}}{2} \left(\frac{1}{\kappa} + \kappa\right)
\]
where the coupling constant is defined as
\[
h_{SQL} = \sqrt{\frac{8 \hbar}{m \Omega^{2} L^{2}}} , \quad
\kappa (\Omega) = \frac{2 \gamma^{4} I_{o}/I_{SQL}}{m L c \Omega^{2} (\gamma^{2} + \Omega^{2})}
\]
and $I_{SQL}$ is the power when $\gamma = \Omega$, defined as
\[
I_{SQL} = \frac{m L^{2} \gamma^{4}}{4 \omega_{o}}
\]
Notice that the laser power $I_{o}$ is the power in the F-P cavities.

### Magnifying at the PRM

The laser power circling in the cavities is different from the laser power input to the detector, mostly because the magnifying effect of the PRM mirror. First we consider the cavity on the arm (ITM and ETM), while steady, the power in and out the cavity at the ITM is
\[
G_{arm} = \frac{P_{in}}{P_{circ}} = \frac{T_{ITM}}{1 - \sqrt{R_{ITM} R_{ETM}}}
\]
Using the approximation $T_{ETM} \ll E_{ITM}$, the ratio is near
\[
\frac{P_{in}}{P_{circ}} \approx \frac{T_{ITM}}{4}
\]
Loss of per circulation is 
\[
\Lambda_{MI} = \frac{4 (T_{ETM} + L_{arm})}{T_{ITM}} + L_{BS}
\]
So the beam splitter and the arm F-P cavity can be regarded as one mirror with equivalent reflection ratio
\[
R_{MI} = 1 - \Lambda_{MI}
\]
Therefore, the gain for the equivalent cavity is
\[
G_{PR} = \frac{4 T_{PRM}}{(T_{PRM} + \Lambda_{MI})^{2}}
\]
Choosing $T_{PRM} = \Lambda_{MI}$, the total gain is approximately 
\[
G_{PR,max} \approx \frac{1}{\Lambda_{MI}}
\]

### Signal Recycling and Detuned Interferometers




[^1]: Sayed Abolfazl Mirdehghan, 1 - Fibrous polymeric composites, Editor(s): Masoud Latifi, In The Textile Institute Book Series, Engineered Polymeric Fibrous Materials, Woodhead Publishing, 2021, Pages 1-58, ISBN 9780128243817, https://doi.org/10.1016/B978-0-12-824381-7.00012-3. (https://www.sciencedirect.com/science/article/pii/B9780128243817000123)
