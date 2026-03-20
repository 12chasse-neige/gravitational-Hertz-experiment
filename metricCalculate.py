import numpy as np
import mpmath as mp
import sys
import argparse

mp.mp.dps = 10  #set the precision for the code

#define the contants that will be used
H = mp.mpf('2')      # height of the column (meter)
D = mp.mpf('5')      # diameter of the column (meter)
d = mp.mpf('1')      # diameter of the holes (meter)
s = mp.mpf('1.5')    # distance from center to holes (meter)
R = mp.mpf('2000')   # distance from source to detector (meter)
rho = mp.mpf('1750') # density (kg/m^3)
G = mp.mpf('6.674e-11')  # gravitational constant
c = mp.mpf('2.998e8')    # speed of light
pi = mp.pi
omega = mp.mpf('300') * mp.mpf('2') * pi  # rotation frequency
sin = mp.sin
cos = mp.cos
def dirac_delta(i, j):
    """
    dirac delta function
    """    
    if i==j:
        return mp.mpf('1')
    else:
        return mp.mpf('0')    
        
def get_metricTensorcomponentTT(r, t):
    """
    describe the rotation of the columns: we assume that at the time t=0, the center of one hole is at the direction of the x axis, and the center of the four holes are defined by x_k and y_k (k = 0,1)
    """
    def get_the_coordinate_of_the_hole(k, t):
        x_k = s * cos(omega * t + k * pi) 
        y_k = s * sin(omega * t + k * pi)
        return x_k, y_k

    #calculate the components of the quadrupole tensor
    #calculate the ij componet of the quadrupole tensor of one column
    # def calculate_one_column_tensor(i, j, D, rho, H):
    #     def y_lower(x):
    #         return -np.sqrt(D**2/4 - x**2)
    #     def y_upper(x):
    #         return np.sqrt(D**2/4 - x**2)
    #     x_lower = -D/2
    #     x_upper = D/2
    #     z_lower = -H/2
    #     z_upper = H/2
    #     def integrand(x, y, z):
    #         coordinates = [x, y, z]
    #         return rho * (coordinates[i] * coordinates[j] - mp.mpf('1')/3 * dirac_delta(i, j) * (x**2 + y**2 + z**2))
        
    #     def f(x):
    #         def g(y):
    #             def h(z):
    #                 return integrand(x, y, z)
    #             return mp.quad(h, [z_lower, z_upper])
    #         return mp.quad(g, [y_lower(x), y_upper(x)])
        
    #     component = mp.quad(f, [x_lower, x_upper])
    #     return component

    def hole_tensor_component(i, j, k, d, rho ,H ,t):
        """
        calculate the componets of quadrupole tensor of the holes relative to the center of the column
        """
        component = mp.mpf('0')
        x = get_the_coordinate_of_the_hole(k, t)[0]
        y = get_the_coordinate_of_the_hole(k, t)[1]
        z = mp.mpf('0')
        def relative_component(i, j, t):
            coordinates = [x, y, z]
            relativeComponent = rho * pi * d**2 / 4 * H * (coordinates[i] * coordinates[j] - mp.mpf('1')/3 * dirac_delta(i, j) * (x**2 + y**2 + z**2))
            return relativeComponent
        component += relative_component(i, j, t)
        return component

    def calculate_the_whole_tensor(i, j, D, rho, H, t):
        """
        adding to get the whole quadrupole tensor I_ij
        """
        big_column = mp.mpf('0')
        component = big_column
        for k in range(2):
            small_column = hole_tensor_component(i, j, k, d, -rho, H, t)
            component += small_column
        return component

    def second_derivative_of_component(i, j, t):
        """
        calculate the second-order derivative of the quadrupole tensor
        """
        if i==2 or j==2:
            return mp.mpf('0')
        else:
            derivative = - mp.mpf('4') * omega ** 2 * calculate_the_whole_tensor(i, j, D, rho, H, t)
            return derivative

    #get the final metric tensor
    t_rev = t - r/c
    metricTensorComponent = [[mp.mpf('0') for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            metricTensorComponent[i][j] = mp.mpf('2') * G * second_derivative_of_component(i, j, t_rev) / (r * c**4)

    def project_to_TT_mode(metricTensorComponent):
        """
        project the mectric tensor to TT mode
        """
        projector = [[mp.mpf('1'), mp.mpf('0'), mp.mpf('0')],
                    [mp.mpf('0'), mp.mpf('1'), mp.mpf('0')],
                    [mp.mpf('0'), mp.mpf('0'), mp.mpf('0')]]
        metricTensorComponentTT = [[mp.mpf('0') for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        metricTensorComponentTT[i][j] += (projector[i][k] * projector[j][l] - mp.mpf('0.5') * projector[i][j] * projector[k][l]) * metricTensorComponent[k][l]
        return metricTensorComponentTT

    #output the metric tensor in TT gauge
    metricTensorComponentTT = project_to_TT_mode(metricTensorComponent)
    return metricTensorComponentTT

#calculate the average respond for monochromatic wave
#define the orientation of the detector
# theta_arm = pi / 2
# phi_arm = pi / 4    #the orientation of one arm of the detector (rad)
# theta_det = mp.mpf('1.57079')
# phi_det = mp.mpf('0.79')   #the orientation of the vector from the centor of the source to the center of the detector

def calculate_deltaT(t, n_i, n_j, n_k, a_i, a_j, a_k, L):
    deltaT = mp.mpf('0')
    def distance(x):    #return the distance from the center of the source to the position of the photon
        return mp.sqrt((R * n_i + x * a_i)**2 + (R * n_j + x * a_j)**2 + (R * n_k + x * a_k)**2)
    for i in range(3):
        for j in range(3):
            deltaT += mp.quad(lambda x: mp.mpf('1') / (2*c) * a_i * a_j * get_metricTensorcomponentTT(distance(x), t + x / c)[i][j], [mp.mpf('0'), L])
    return deltaT

def calculate_deltaTPrime(t, n_i, n_j, n_k, a_i, a_j, a_k, L):
    deltaTPrime = mp.mpf('0')
    def distance(x):    #return the distance from the center of the source to the position of the photon
        return mp.sqrt((R * n_i + x * a_i)**2 + (R * n_j + x * a_j)**2 + (R * n_k + x * a_k)**2)
    for i in range(3):
        for j in range(3):
            deltaTPrime += mp.quad(lambda x: mp.mpf('1') / (2*c) * a_i * a_j * get_metricTensorcomponentTT(distance(x), t + (L - x) / c)[i][j], [mp.mpf('0'), L])
    return deltaTPrime

def parse_arguments():
    """
    Returns:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        epilog='example: python metricCalculate.py time theta_arm phi_arm theta_det phi_det data.txt'
    )
    
    parser.add_argument(
        '-t', '--time',
        type=str,
        required=True,
        help='current time'
    )

    parser.add_argument(
        '-ta', '--thetaarm',
        type=str,
        required=True,
        help='position angle of the detector arm'
    )

    parser.add_argument(
        '-pa', '--phiarm',
        type=str,
        required=True,
        help='position angle of the detector arm'
    )

    parser.add_argument(
        '-td', '--thetadetector',
        type=str,
        required=True,
        help='position angle of the detector'
    )

    parser.add_argument(
        '-pd', '--phidetector',
        type=str,
        required=True,
        help='position angle of the detector'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='show detailed output'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='path for the output file'
    )
    
    return parser.parse_args()

def main(t = float, theta_arm  = 1.6882, phi_arm = 0.7850, theta_det = 0.0000, phi_det = 1.1981):

    a_i, a_j, a_k = sin(theta_arm) * cos(phi_arm), sin(theta_arm) * sin(phi_arm), cos(theta_arm)
    n_i, n_j, n_k = sin(theta_det) * cos(phi_det), sin(theta_det) * sin(phi_det), cos(theta_det)
    L = mp.mpf('1000')    #length of the arm of the detector (meter)

    delay_of_transition_time = calculate_deltaT(t - 2 * L / c, n_i, n_j, n_k, a_i, a_j, a_k, L) + calculate_deltaTPrime(t - L / c, n_i, n_j, n_k, a_i, a_j, a_k, L)
    input_signal = delay_of_transition_time * c / (2 * L)

    return(input_signal)


if __name__ == "__main__":

    args = parse_arguments()

    if args.verbose:
        print(f"begin processing, current time: {args.time}")
        if args.output:
            print(f"output file: {args.output}")
    
    time = float(args.time)
    theta_arm = float(args.thetaarm)
    phi_arm = float(args.phiarm)
    theta_det = float(args.thetadetector)
    phi_det = float(args.phidetector)
    result = main(time, theta_arm, phi_arm, theta_det, phi_det)
    print(result)
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(str(result))
            print(f"\nresult saved at: {args.output}")
        except Exception as e:
            print(f"error while saving file: {e}")
    
    sys.exit(0)